import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np
import math
import sys
import pdb
# from udlp.utils import Dataset, masking_noise
# from udlp.ops import MSELoss, BCELoss
from lib.opsLatentTree_pyltm import LatentTreeModule
from lib.utils import Dataset, saveToBif,log_sum_exp
import logging
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import cluster, datasets, mixture

def cluster_accuracy(assigments, labels):
    """
    Assume labels are from 0 - K-1
    """
    clusters = np.unique(assigments)
    classes = np.unique(labels)

    num_hit = 0
    for c in clusters:
        subassignments = assigments[assigments==c]
        sublabels = labels[assigments==c]
        counts = np.zeros(len(classes))
        for l in range(len(classes)):
            counts[l] = np.sum(sublabels==classes[l])
        cluster_label = classes[np.argmax(counts)]
        num_hit += np.sum(sublabels==cluster_label)

    # ave_acc = ave_acc/(len(clusters))
    ave_acc = 1.0*num_hit/len(assigments)
    return ave_acc

def _estimate_gaussian_parameters(X, resp, reg_covar=1e-6):
    '''resp: responsibility of each data point to each of the K clusters
    i.e. posterior p(z=k|x)
    '''
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    covariances = avg_X2 - avg_means2 + reg_covar

    return nk, means, covariances

def buildNetwork(layers, activation="relu", dropout=0.0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

def adjust_learning_rate(init_lr, epoch):
    lr = init_lr * (0.9 ** (epoch//10))
    return lr

log2pi = math.log(2*math.pi)
def log_likelihood_samplesImean_sigma(samples, mu, logvar):
    return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples-mu)**2/torch.exp(logvar) + 0.5*logvar, 1)

class LTVAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
        encodeLayer=[400], decodeLayer=[400], alpha=1.):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.alpha = alpha
        self.encoder = buildNetwork([input_dim] + encodeLayer)
        self.decoder = buildNetwork([z_dim] + decodeLayer)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

        self.latentTree = LatentTreeModule()

    def updateLatentTree(self, model_file, varNames):
        self.latentTree.updateLatentTree(model_file, varNames)
        self.latentTree.latObj.setCovariances(self.alpha)

    def initialize(self, X, y, n_components, path, netName, varNames, n_init=10, random_state=None):
        n_samples, _ = X.shape

        resp = np.zeros((n_samples, n_components))
        label = cluster.KMeans(n_clusters=n_components, n_init=n_init,
                               random_state=random_state).fit(X).labels_
        resp[np.arange(n_samples), label] = 1
        logging.info("Initial # cluster: %d, acc: %.5f, nmi: %.5f" % (n_components, cluster_accuracy(label, y), 
                    normalized_mutual_info_score(y, label)))

        n_samples, _ = X.shape
        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp)
        weights /= n_samples
        covariances = np.ones_like(means)*self.alpha
        saveToBif(path, netName, varNames, weights, means, covariances)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu
        # return mu + 0

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x, z, mu, logvar):
        if self._dec_act is not None:
            dataloglikelihood = torch.mean(torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1))
        else:
            dataloglikelihood = -torch.mean(torch.sum((recon_x-x)**2, 1))

        zLatentTree = z * 1
        # zLatentTree.register_hook(print)
        zloglikelihood = torch.mean(self.latentTree(zLatentTree))
        # print(logvar)
        qentropy = 0.5*torch.mean(torch.sum(1+logvar+math.log(2 * math.pi), 1))
        elbo = dataloglikelihood + zloglikelihood + qentropy
        loss = -elbo
        return loss, dataloglikelihood, zloglikelihood, qentropy

    def forward(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    def encodeBatch(self, dataloader, islabel=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        ylabels = []
        self.eval()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z,_,_,_ = self.forward(inputs)
            encoded.append(z.data.cpu())
            ylabels.append(labels)

        encoded = torch.cat(encoded, dim=0)
        ylabels = torch.cat(ylabels)
        if islabel:
            out = (encoded, ylabels)
        else:
            out = encoded
        return out

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def evaluate(self, dataloader):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        self.eval()
        valid_loss = 0.0
        data_log = 0.
        z_log = 0.
        qentropy_log = 0.
        latents = None
        ylabels = None
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs, mu, logvar = self.forward(inputs)

            loss, dataloglikelihood, zloglikelihood, qentropy = self.loss_function(outputs, inputs, z, mu, logvar)
            valid_loss += loss.data*len(inputs)
            data_log += dataloglikelihood.data*len(inputs)
            z_log += zloglikelihood.data*len(inputs)
            qentropy_log += qentropy.data*len(inputs)

            x = z.data.cpu().numpy()
            y = labels.cpu().numpy()
            latent = self.latentTree.latObj.inference(x)
            if batch_idx==0:
                latents = latent
                ylabels = y
            else:
                for j in range(len(latent)):
                    latents[j] = np.concatenate((latents[j], latent[j]), axis=0)
                ylabels = np.concatenate((ylabels, y))

        logging.info("Valid Loss: %.5f, dataloglikelihood: %.5f, zloglikelihood: %.5f, qentropy: %.5f" % (
                valid_loss / len(dataloader.dataset), -data_log/len(dataloader.dataset),
                -z_log/len(dataloader.dataset), -qentropy_log/len(dataloader.dataset)))
        for j in range(len(latents)):
            latent = latents[j]
            num, cluster = latent.shape
            assigments = np.argmax(latent, axis=1)
            num_facets = 1
            if len(ylabels.shape)>1:
                num_facets = ylabels.shape[1]
            if num_facets>1:
                for i in range(num_facets):
                    y = ylabels[:, i]
                    logging.info("Facet %d -> gt facet %d, # cluster: %d, acc: %.5f, nmi: %.5f" % (j, i, cluster, cluster_accuracy(assigments, y), 
                        normalized_mutual_info_score(y, assigments)))
            else:
                logging.info("Facet %d # cluster: %d, acc: %.5f, nmi: %.5f" % (j, cluster, cluster_accuracy(assigments, ylabels), 
                    normalized_mutual_info_score(ylabels, assigments)))

    def fit(self, trainloader, testloader, lr=0.001, stepwise_em_lr = 0.001, batch_size=128, num_epochs=10,
        visualize=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        # self.evaluate(trainloader)
        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            train_loss = 0.
            data_log = 0.
            z_log = 0.
            qentropy_log = 0.
            stepwise_em_lr_annealed = adjust_learning_rate(stepwise_em_lr, epoch)
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                z, outputs, mu, logvar = self.forward(inputs)
                # pdb.set_trace()
                # z.register_hook(print)
                loss, dataloglikelihood, zloglikelihood, qentropy = self.loss_function(outputs, inputs, z, mu, logvar)
                train_loss += loss.data*len(inputs)
                data_log += dataloglikelihood.data*len(inputs)
                z_log += zloglikelihood.data*len(inputs)
                qentropy_log += qentropy.data*len(inputs)
                loss.backward()
                optimizer.step()

                # stepwise em
                self.latentTree.latObj.stepwise_em_step(z.data.cpu().numpy(), stepwise_em_lr, batch_size, updatevar=False)


            logging.info("#Epoch %3d: Train Loss: %.5f, dataloglikelihood: %.5f, zloglikelihood: %.5f, qentropy: %.5f" % (
                epoch, train_loss / len(trainloader.dataset), -data_log/len(trainloader.dataset),
                -z_log/len(trainloader.dataset), -qentropy_log/len(trainloader.dataset)))
            if (epoch+1) % 10 == 0 or epoch==num_epochs-1:
                self.evaluate(trainloader)

            if visualize:
                sample = Variable(torch.randn(64, self.z_dim))
                if use_cuda:
                   sample = sample.cuda()
                sample = self.decode(sample).cpu()
                save_image(sample.data.view(64, 1, 28, 28),
                           'results/ltvae/sample/sample_' + str(epoch) + '.png')

    def log_marginal_likelihood_estimate(self, x, num_samples):
        weight = torch.zeros(x.size(0), num_samples)
        for i in range(num_samples):
            z, recon_x, mu, logvar = self.forward(x)
            zloglikelihood = self.latentTree(z)
            if self._dec_act is not None:
                dataloglikelihood = torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                    (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
            else:
                dataloglikelihood = -torch.mean(torch.sum((recon_x-x)**2, 1))
            log_qz = log_likelihood_samplesImean_sigma(z, mu, logvar)
            weight[:, i] = (dataloglikelihood + zloglikelihood - log_qz).data.cpu()
        # pdb.set_trace()
        return log_sum_exp(weight, dim=1) - math.log(num_samples)
