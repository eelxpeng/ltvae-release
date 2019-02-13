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
import logging
from lib.utils import Dataset, acc, log_sum_exp
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

log2pi = math.log(2*math.pi)
def log_likelihood_samplesImean_sigma(samples, mu, logvar):
    return -0.5*log2pi*samples.size()[1] - torch.sum(0.5*(samples-mu)**2/torch.exp(logvar) + 0.5*logvar, 1)

def _estimate_gaussian_parameters(X, resp, reg_covar):
    '''resp: responsibility of each data point to each of the K clusters
    i.e. posterior p(z=k|x)
    '''
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means ** 2
    covariances = avg_X2 - avg_means2 + reg_covar

    return nk, means, covariances

def _compute_precision_cholesky(covariances):
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")
    if np.any(np.less_equal(covariances, 0.0)):
        raise ValueError(estimate_precision_error_message)
    precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol

def _estimate_log_gaussian_prob(X, means, precisions_chol):
    n_components, n_features = means.shape
    log_det = (np.sum(np.log(precisions_chol), axis=1))
    precisions = precisions_chol ** 2
    log_prob = (np.sum((means ** 2 * precisions), 1) -
                2. * np.dot(X, (means * precisions).T) +
                np.dot(X ** 2, precisions.T))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

def logsumexp(a, axis=None, keepdims=False):
    a_max = np.amax(a, axis=axis, keepdims=True)
    tmp = np.exp(a - a_max)
    s = np.sum(tmp, axis=axis, keepdims=keepdims)
    out = np.log(s)
    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max
    return out

def adjust_learning_rate(init_lr, epoch, optimizer=None):
    lr = init_lr * (0.9 ** (epoch//20))
    if optimizer is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr

class GMMVAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_components=10, binary=True,
        encodeLayer=[400], decodeLayer=[400], activation="relu", 
        dropout=0, tied=False, alpha=1., reg_covar=1e-6, max_iter=100,
        random_state=None, tol=1e-3, init_params='kmeans'):
        super(self.__class__, self).__init__()
        self.init_params = init_params
        self.max_iter = max_iter
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.tol = tol

        self.z_dim = z_dim
        self.n_components = n_components
        self.alpha = alpha
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

        self.weights_ = np.ones(n_components) / n_components
        self.means_ = np.zeros((n_components, z_dim))
        self.covariances_ = np.ones((n_components, z_dim))*self.alpha
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_)

    def gmmfit(self, X, max_iter=None, init=False):
        if init:
            self._initialize(X)
        self.lower_bound_ = -np.infty
        self.converged_ = False
        for n_iter in range(self.max_iter):
            prev_lower_bound = self.lower_bound_
            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            self.lower_bound_ = log_prob_norm

            change = self.lower_bound_ - prev_lower_bound
            if abs(change) < self.tol:
                self.converged_ = True
                break

        if not self.converged_:
            warnings.warn('Initialization did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.')
        logging.info("Averaged loglikelihood: %f, n_iter: %d" % (self.score(X), n_iter))

    def score(self, X):
        return logsumexp(self._estimate_weighted_log_prob(X), axis=1).mean()

    def _initialize(self, X, n_init=10):
        n_samples, _ = X.shape

        if self.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=n_init,
                                   random_state=self.random_state).fit(X).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            resp = self.random_state.rand(n_samples, self.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.init_params)

        n_samples, _ = X.shape
        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar)
        weights /= n_samples

        self.weights_ = weights
        self.means_ = means
        # self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(
            covariances)


    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        # self.weights_, self.means_, self.covariances_ = (
        #     _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar))
        self.weights_, self.means_, _ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_)
        
    def log_prob(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return log_prob_norm

    def _estimate_log_prob_resp(self, X):
        """
        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def predict(self, X):
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def stepwise_e_step(self, batchX):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(batchX)
        return np.mean(log_prob_norm), log_resp

    def stepwise_m_step(self, batchX, log_resp, learning_rate, batch_size):
        n_samples, _ = batchX.shape
        assert(n_samples==batch_size)
        resp = np.exp(log_resp)
        # initialize sufficient statistics for batch_size of data
        sufficient_counts = batch_size * self.weights_      # (K, )
        sufficient_counts = np.expand_dims(sufficient_counts, axis=1) # (K, 1)
        sufficient_sum = self.means_ * sufficient_counts    # (K, D)
        sufficient_sum_square = (self.covariances_ + self.means_**2) * sufficient_counts

        # update sufficient statistics
        sufficient_counts = sufficient_counts + learning_rate * (
            np.sum(resp, axis=0, keepdims=True).T - sufficient_counts)
        resp_3d = np.expand_dims(resp, axis=1) # Nx1xK
        batchX_3d = np.expand_dims(batchX, axis=2) # NxDx1
        sufficient_sum = sufficient_sum + learning_rate * (
            np.sum(resp_3d * batchX_3d, axis=0).T - sufficient_sum)
        sufficient_sum_square = sufficient_sum_square + learning_rate * (
            np.sum(resp_3d * batchX_3d**2, axis=0).T - sufficient_sum_square)

        # update parameters
        self.weights_ = np.squeeze(sufficient_counts, axis=1) / batch_size
        self.means_ = sufficient_sum / sufficient_counts
        # self.covariances_ = sufficient_sum_square / sufficient_counts - self.means_**2

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_)
        
    def stepwise_em_step(self, batchX, learning_rate, batch_size):
        log_prob_norm, log_resp = self.stepwise_e_step(batchX)
        self.stepwise_m_step(batchX, log_resp, learning_rate, batch_size)
        return log_resp

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

    def loss_function(self, recon_x, x, z, mu, logvar, log_resp, u_p, lambda_p):
        if self._dec_act is not None:
            dataloglikelihood = torch.mean(torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1))
            # dataloglikelihood = torch.mean(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                # (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)))
        else:
            dataloglikelihood = -torch.mean(torch.sum((recon_x-x)**2, 1))
        z = z * 1
        # pdb.set_trace()
        # z.register_hook(print)
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_components) # NxDxK
        u_tensor3 = u_p.unsqueeze(0).expand(z.size()[0], u_p.size()[0], u_p.size()[1]) # NxDxK
        lambda_tensor3 = lambda_p.unsqueeze(0).expand(z.size()[0], lambda_p.size()[0], lambda_p.size()[1])
        resp_3d = torch.exp(log_resp).unsqueeze(1) # Nx1xK
        zloglikelihood = -torch.mean(torch.sum(torch.sum(resp_3d * (Z - u_tensor3)**2 * lambda_tensor3 * 0.5, 1), 1))
        # print(logvar)
        qentropy = 0.5*torch.mean(torch.sum(1+logvar+math.log(2 * math.pi), 1))
        # elbo = dataloglikelihood + self.alpha*(zloglikelihood + qentropy)
        elbo = dataloglikelihood + zloglikelihood + qentropy
        loss = -elbo
        return loss, -dataloglikelihood, -zloglikelihood, -qentropy

    def forward(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z, self.decode(z), mu, logvar

    def encodeBatch(self, data):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            data = data.cuda()
        z, _, mu, _ = self.forward(data)
        return mu.data.cpu()

    def save_model(self, path):
        tensorfile = path+".pt"
        numpyfile = path+".npz"
        torch.save(self.state_dict(), tensorfile)
        np.savez(numpyfile, weights=self.weights_, means=self.means_, 
            covariances=self.covariances_)

    def load_pretrain(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def load_model(self, path):
        tensorfile = path+".pt"
        numpyfile = path+".npz"
        pretrained_dict = torch.load(tensorfile, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

        param = np.load(numpyfile)
        self.weights_ = param["weights"]
        self.means_ = param["means"]
        self.covariances_ = param["covariances"]
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_)

    def learn_gmm(self, dataset, max_iter=200, init=False):
        dataX, dataY = dataset[:]
        X = self.encodeBatch(dataX).cpu().numpy()
        Y = dataY.cpu().numpy()
        self.gmmfit(X, max_iter=max_iter, init=init)
        y_pred = self.predict(X)
        logging.info("acc: %.5f, nmi: %.5f" % (acc(Y, y_pred), normalized_mutual_info_score(Y, y_pred)))

        log_prob_norm, log_resp = self._e_step(X)
        return log_resp

    def evaluate(self, dataset):
        dataX, dataY = dataset[:]
        X = self.encodeBatch(dataX).cpu().numpy()
        Y = dataY.cpu().numpy()
        y_pred = self.predict(X)
        accuracy = acc(Y, y_pred)
        nmi = normalized_mutual_info_score(Y, y_pred)
        logging.info("acc: %.5f, nmi: %.5f" % (accuracy, nmi))
        return accuracy, nmi

    def fit(self, trainloader, validloader, lr=0.001, lr_stepwise=0.01, batch_size=128, num_epochs=10,
        anneal=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        logging.info("=====Initialize Gaussian Mixtures=======")
        log_resp = self.learn_gmm(trainloader.dataset, init=True)

        logging.info("=====Finetuning=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        lr_stepwise_epoch = lr_stepwise

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            if anneal:
                adjust_learning_rate(lr, epoch, optimizer)
                lr_stepwise_epoch = adjust_learning_rate(lr_stepwise, epoch)

            train_loss = 0
            train_recon_loss = 0.0
            train_cluster_loss = 0.0
            train_entropy_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                # log_resp_batch_tensor = torch.from_numpy(log_resp_batch).float()
                u_p = torch.from_numpy(self.means_).float().t()
                lambda_p = torch.from_numpy(self.precisions_cholesky_**2).float().t()

                if use_cuda:
                    inputs = inputs.cuda()
                    # log_resp_batch_tensor = log_resp_batch_tensor.cuda()
                    u_p = u_p.cuda()
                    lambda_p = lambda_p.cuda()

                optimizer.zero_grad()
                inputs = Variable(inputs)
                # log_resp_batch_tensor = Variable(log_resp_batch_tensor)
                
                z, outputs, mu, logvar = self.forward(inputs)
                # z.register_hook(print)
                
                _, log_resp_batch = self.stepwise_e_step(z.data.cpu().numpy())
                log_resp_batch_tensor = torch.from_numpy(log_resp_batch).float()
                if use_cuda:
                    log_resp_batch_tensor = log_resp_batch_tensor.cuda()
                log_resp_batch_tensor = Variable(log_resp_batch_tensor)
                    
                loss, dataloglikelihood, zloglikelihood, qentropy = self.loss_function(
                    outputs, inputs, z, mu, logvar, log_resp_batch_tensor, u_p, lambda_p)
                
                train_loss += loss.data*len(inputs)
                train_recon_loss += dataloglikelihood.data*len(inputs)
                train_cluster_loss += zloglikelihood.data*len(inputs)
                train_entropy_loss += qentropy.data*len(inputs)
                loss.backward()
                optimizer.step()

                # Perform mini-batch stepwise EM
                temp_log_resp = self.stepwise_em_step(
                    z.data.cpu().numpy(), lr_stepwise_epoch, len(inputs))
                # log_resp[batch_idx*batch_size : min((batch_idx+1)*batch_size, num_train)] = temp_log_resp

            num_train = len(trainloader.dataset)
            logging.info("#Epoch %3d: Loss: %.3f, Recon Loss: %.3f, Cluster Loss: %.3f, Entropy Loss: %.3f" % (
                epoch+1, train_loss / num_train, train_recon_loss/num_train, train_cluster_loss/num_train,
                train_entropy_loss / num_train))

            if (epoch+1) % 10 == 0:
                    self.evaluate(validloader.dataset)

    def log_marginal_likelihood_estimate(self, x, num_samples):
        weight = torch.zeros(x.size(0), num_samples)
        use_cuda = torch.cuda.is_available()
        for i in range(num_samples):
            z, recon_x, mu, logvar = self.forward(x)
            zloglikelihood = torch.from_numpy(self.log_prob(z.data.cpu().numpy())).float()
            if use_cuda:
                zloglikelihood = zloglikelihood.cuda()
            if self._dec_act is not None:
                dataloglikelihood = torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                    (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
            else:
                dataloglikelihood = -torch.sum((recon_x-x)**2, 1)
            log_qz = log_likelihood_samplesImean_sigma(z, mu, logvar)
            weight[:, i] = (dataloglikelihood + zloglikelihood - log_qz).data.cpu()
            # pdb.set_trace()
        return log_sum_exp(weight, dim=1) - math.log(num_samples)
        # return torch.log(weight/num_samples)

