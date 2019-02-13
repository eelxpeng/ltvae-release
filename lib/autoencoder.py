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
# from lib.ops import MSELoss, BCELoss
from sklearn.mixture import GaussianMixture
from lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

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

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
        encodeLayer=[400], decodeLayer=[400]):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.encoder = buildNetwork([input_dim] + encodeLayer)
        self.decoder = buildNetwork([z_dim] + decodeLayer)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

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
            z,_ = self.forward(inputs)
            encoded.append(z.data)
            ylabels.append(labels)

        encoded = torch.cat(encoded, dim=0)
        ylabels = torch.cat(ylabels)
        if islabel:
            out = (encoded, ylabels)
        else:
            out = encoded
        return out

    def loss_function(self, recon_x, x):
        if self._dec_act is not None:
            loss = -torch.mean(torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
                (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1))
            # loss = F.binary_cross_entropy(recon_x, x)
        else:
            loss = torch.mean(torch.sum((recon_x-x)**2, 1))
            # loss = F.mse_loss(recon_x, x)

        return loss

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)

        return z, self.decode(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, 
        visualize=False, anneal=False, optimizer="adam"):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        if optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        # validate
        self.eval()
        valid_loss = 0.0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs = self.forward(inputs)

            loss = self.loss_function(outputs, inputs)
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
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                
                z, outputs = self.forward(inputs)
                loss = self.loss_function(outputs, inputs)
                train_loss += loss.data*len(inputs)
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            # validate
            self.eval()
            valid_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs = self.forward(inputs)

                loss = self.loss_function(outputs, inputs)
                valid_loss += loss.data*len(inputs)

            print("#Epoch %3d: Train Loss: %.5f, Valid Loss: %.5f" % (
                epoch, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))

            if epoch % int(num_epochs/10) == 0 or epoch==num_epochs-1:
                trainX, trainY = self.encodeBatch(trainloader, True)
                testX, testY = self.encodeBatch(validloader, True)
                trainX = trainX.cpu().numpy()
                trainY = trainY.cpu().numpy()
                testX = testX.cpu().numpy()
                testY = testY.cpu().numpy()
                n_components = len(np.unique(trainY))
                km = KMeans(n_clusters=n_components, n_init=20).fit(trainX)
                y_pred = km.predict(testX)
                print("acc: %.5f, nmi: %.5f" % (acc(testY, y_pred), normalized_mutual_info_score(testY, y_pred)))
                gmm = GaussianMixture(n_components=n_components, covariance_type='diag', means_init=km.cluster_centers_).fit(trainX)
                y_pred = gmm.predict(testX)
                print("acc: %.5f, nmi: %.5f" % (acc(testY, y_pred), normalized_mutual_info_score(testY, y_pred)))
            