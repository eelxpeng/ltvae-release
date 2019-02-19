'''
Created on 26 Sep 2018

@author: Bryan
'''
import numpy as np
from sklearn import cluster, datasets, mixture
import math

from pyltm.learner import EMFramework
from pyltm.data import ContinuousDatacase
from pyltm.reasoner import NaturalCliqueTreePropagation, Evidence
from pyltm.model.node.continuous_belief_node import ContinuousBeliefNode

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

class FullEM(object):
    '''
    Assuming no pouch nodes
    '''


    def __init__(self, net):
        '''
        Constructor
        '''
        self._model = net
    
    def initialize(self, evidence):
        for node in self._model.getInternalNodes():
            continuousChildNodes = []
            for child in node.children:
                if isinstance(child, ContinuousBeliefNode):
                    continuousChildNodes.append(child)
            if len(continuousChildNodes)==0:
                continue
            variables = []
            for child in continuousChildNodes:
                assert(len(child.variable.variables)==1)
                variables += child.variable.variables
            data = evidence.getValues(variables)
            n_components = node.variable.getCardinality()
            means, covariances = self.init_gaussian_mixture(data, n_components)
            for i in range(len(continuousChildNodes)):
                child = continuousChildNodes[i]
                for k in range(n_components):
                    child.potential.get(k).mu[:] = means[k, i]
                    child.potential.get(k).covar[:] = covariances[k, i]
            
    def init_gaussian_mixture(self, X, n_components, n_init=10):
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, n_components))
        label = cluster.KMeans(n_clusters=n_components, n_init=n_init).fit(X).labels_
        resp[np.arange(n_samples), label] = 1
        _, means, covariances = _estimate_gaussian_parameters(
            X, resp)
        return means, covariances
        
    def fit(self, data, varNames, learning_rate, batch_size, num_epochs, initialize=True):
        if initialize:
            # set up evidence
            datacase = ContinuousDatacase.create(varNames)
            datacase.synchronize(self._model)
            datacase.putValues(data)
            evidence = datacase.getEvidence()
            self.initialize(evidence)
        em = EMFramework(self._model, batch_size)
        for epoch in range(num_epochs):
            num_iters = int(math.floor(len(data))/batch_size)
            loglikelihood = 0.0
            for i in range(num_iters):
                batchX = data[i*batch_size:min((i+1)*batch_size, len(data))]
                batchloglikelihood = em.stepwise_em_step(batchX, varNames, learning_rate)
                loglikelihood += np.sum(batchloglikelihood)
            print("# Epoch %d: loglikelihood=%.5f" % (epoch, loglikelihood/(num_iters*batch_size)))
            
        
        