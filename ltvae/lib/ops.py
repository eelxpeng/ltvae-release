import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask=None):
        "mask is a tensor"
        super(self.__class__, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.bias_in = Parameter(torch.Tensor(in_features))
        if mask is not None:
            self.mask = Parameter(mask, requires_grad=False)
        else:
            mask = torch.ones(out_features, in_features)
            self.mask = Parameter(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.bias_in.size(0))
        self.bias_in.data.uniform_(-stdv, stdv)

    def forward(self, input):
        maskedWeight = self.weight*self.mask.t()
        return F.linear(input, maskedWeight, self.bias)

    def decode(self, hidden):
        maskedWeight = self.weight*self.mask.t()
        recon = F.linear(hidden, maskedWeight.t(), self.bias_in)
        return recon

    def get_weight(self):
        return self.weight*self.mask.t()

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class ClassifyLayer(nn.Module):
    def __init__(self, in_features, num_classes, keep_prob=0.8):
        super(self.__class__, self).__init__()
        self.dropout = nn.Dropout(keep_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        # return torch.sqrt(torch.mean((input - target)**2))
        return torch.mean(torch.sum((input-target)**2, 1))

class BCELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return -torch.mean(torch.sum(target*torch.log(torch.clamp(input, min=1e-10))+
            (1-target)*torch.log(torch.clamp(1-input, min=1e-10)), 1))

