'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from PIL import Image

import logging

def weights_xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)
        
def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)

class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def saveToArff(data, name, varNames, outpath):
    def writePreamble(writer, relationName, varNames):
        writer.write("@relation %s\n" % relationName)
        for var in varNames:
            writer.write("@attribute %s REAL\n" % var)

    def writeInstances(writer, data):
        writer.write("@data\n")
        for d in data:
            writer.write("%s\n" % (",".join([str(x) for x in d])))

    with open(outpath, "w") as writer:
        writePreamble(writer, name, varNames)
        writeInstances(writer, data)

def saveToBif(path, netName, varNames, weights, means, covariances):
    def writeNetworkDeclaration(netName, writer):
        writer.write('network "%s" {\n}\n' % netName)
        writer.write('\n')

    def writeContinuousVariables(varNames, writer):
        for variable in varNames:
            writer.write('variable "%s" {\n' % variable)
            writer.write('\ttype continuous;\n')
            writer.write('}\n\n')

    def writeDiscreteVariable(varName, num_states, writer):
        states = ['state%d' % state for state in range(num_states)]
        writer.write('variable "%s" {\n' % varName)
        writer.write('\ttype discrete[%d] { ' % num_states)
        for state in states:
            writer.write('"%s" ' % state)
        writer.write("};\n")
        writer.write('}\n\n')

    def writeRootProbabilities(varName, weights, writer):
        writer.write("probability (")
        writer.write('"%s"' % varName)
        writer.write(') {\n')
        writer.write("\ttable ");
        writer.write(" ".join([str(x) for x in weights]))
        writer.write(";\n")
        writer.write('}\n\n')

    def writeConditionalProbabilities(parName, varNames, means, covariances):
        k, d = means.shape
        for j in range(d):
            writer.write("probability (")
            writer.write('"%s"' % varNames[j])
            writer.write(" | ")
            writer.write('"%s"' % parName)
            writer.write(') {\n')
            for i in range(k):
                writer.write('\t("state%d") ' % i)
                writer.write(str(means[i, j]))
                writer.write(' ')
                writer.write(str(covariances[i, j]))
                writer.write(';\n')
            writer.write('}\n\n')

    k, d = means.shape
    with open(path, "w") as writer:
        writeNetworkDeclaration(netName, writer)
        writeDiscreteVariable("y", k, writer)
        writeContinuousVariables(varNames, writer)
        writeRootProbabilities("y", weights, writer)
        writeConditionalProbabilities("y", varNames, means, covariances)
        
def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise

def cluster_accuracy(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

  
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)