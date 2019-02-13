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
# from udlp.utils import Dataset, masking_noise
# from udlp.ops import MSELoss, BCELoss
from sklearn.mixture import GaussianMixture
from lib.utils import acc, log_sum_exp
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import pdb

def buildNetwork(layers, activation="relu", dropout=0):
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

def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.9 ** (epoch//10))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

log2pi = math.log(2*math.pi)
def log_likelihood_samples_unit_gaussian(samples):
    return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples)**2, 1)

def log_likelihood_samplesImean_sigma(samples, mu, logvar):
    return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples-mu)**2/torch.exp(logvar) + 0.5*logvar, 1)

class VAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
        encodeLayer=[400], decodeLayer=[400], alpha=1):
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

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x, mu, logvar):
        if self._dec_act is not None:
            BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
        else:
            BCE = torch.sum((recon_x-x)**2, 1)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        # Normalise by same number of elements as in reconstruction
        recon_loss = torch.mean(BCE)
        kl_loss = torch.mean(KLD)
        loss = self.alpha*recon_loss + kl_loss

        return loss, recon_loss, kl_loss

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
            ylabels.append(labels.cpu())

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

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, 
        visualize=False, anneal=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        # validate
        self.eval()
        valid_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs, mu, logvar = self.forward(inputs)

            loss, _, _ = self.loss_function(outputs, inputs, mu, logvar)
            valid_loss += loss.data*len(inputs)
            # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
            # total_num += inputs.size()[0]

        # valid_loss = total_loss / total_num
        print("#Epoch -1: Valid Loss: %.5f" % (valid_loss / len(validloader.dataset)))

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            if anneal:
                adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0
            recon_loss = 0
            kl_loss = 0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                z, outputs, mu, logvar = self.forward(inputs)
                loss, rl, kl = self.loss_function(outputs, inputs, mu, logvar)
                train_loss += loss.data*len(inputs)
                recon_loss += rl.data*len(inputs)
                kl_loss += kl.data*len(inputs)
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            # validate
            self.eval()
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs, mu, logvar = self.forward(inputs)

                loss, _, _ = self.loss_function(outputs, inputs, mu, logvar)
                valid_loss += loss.data*len(inputs)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]

                # view reconstruct
                if visualize and batch_idx == 0:
                    n = min(inputs.size(0), 8)
                    comparison = torch.cat([inputs.view(-1, 1, 28, 28)[:n],
                                            outputs.view(-1, 1, 28, 28)[:n]])
                    save_image(comparison.data.cpu(),
                                 'results/vae/reconstruct/reconstruction_' + str(epoch) + '.png', nrow=n)

            # valid_loss = total_loss / total_num
            print("#Epoch %3d: Train Loss: %.2f, Recon Loss: %.2f, KL Loss: %.2f, Valid Loss: %.2f" % (
                epoch, train_loss / len(trainloader.dataset), 
                recon_loss / len(trainloader.dataset), 
                kl_loss / len(trainloader.dataset), 
                valid_loss / len(validloader.dataset)))

            if epoch % int(num_epochs/10) == 0 or epoch==num_epochs-1:
                trainX, trainY = self.encodeBatch(trainloader, True)
                testX, testY = self.encodeBatch(validloader, True)
                trainX = trainX.numpy()
                trainY = trainY.numpy()
                testX = testX.numpy()
                testY = testY.numpy()
                n_components = len(np.unique(trainY))
                km = KMeans(n_clusters=n_components, n_init=20).fit(trainX)
                y_pred = km.predict(testX)
                print("acc: %.5f, nmi: %.5f" % (acc(testY, y_pred), normalized_mutual_info_score(testY, y_pred)))
                gmm = GaussianMixture(n_components=n_components, covariance_type='diag', means_init=km.cluster_centers_).fit(trainX)
                y_pred = gmm.predict(testX)
                print("acc: %.5f, nmi: %.5f" % (acc(testY, y_pred), normalized_mutual_info_score(testY, y_pred)))
            
            # view sample
            if visualize:
                sample = Variable(torch.randn(64, self.z_dim))
                if use_cuda:
                   sample = sample.cuda()
                sample = self.decode(sample).cpu()
                save_image(sample.data.view(64, 1, 28, 28),
                           'results/vae/sample/sample_' + str(epoch) + '.png')

    # def log_marginal_likelihood_estimate(self, x, num_samples):
    #     weight = torch.zeros(x.size(0))
    #     for i in range(num_samples):
    #         z, recon_x, mu, logvar = self.forward(x)
    #         zloglikelihood = log_likelihood_samples_unit_gaussian(z)
    #         if self._dec_act is not None:
    #             dataloglikelihood = torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
    #                 (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
    #         else:
    #             dataloglikelihood = -torch.sum((recon_x-x)**2, 1)
    #         log_qz = log_likelihood_samplesImean_sigma(z, mu, logvar)
    #         weight += torch.exp(dataloglikelihood + zloglikelihood - log_qz).data.cpu()
    #         # pdb.set_trace()
    #     return torch.log(torch.clamp(weight/num_samples, min=1e-40))
    def log_marginal_likelihood_estimate(self, x, num_samples):
        weight = torch.zeros(x.size(0), num_samples)
        for i in range(num_samples):
            z, recon_x, mu, logvar = self.forward(x)
            zloglikelihood = log_likelihood_samples_unit_gaussian(z)
            if self._dec_act is not None:
                dataloglikelihood = torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                    (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
            else:
                dataloglikelihood = -torch.sum((recon_x-x)**2, 1)
            log_qz = log_likelihood_samplesImean_sigma(z, mu, logvar)
            weight[:, i] = (dataloglikelihood + zloglikelihood - log_qz).data.cpu()
            # pdb.set_trace()
        return log_sum_exp(weight, dim=1) - math.log(num_samples)

