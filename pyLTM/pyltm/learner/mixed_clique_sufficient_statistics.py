'''
Created on 12 Sep 2018

@author: Bryan
'''
from .sufficient_statistics import SufficientStatistics
from pyltm.model.potential.cgpotential import CGPotential
from pyltm.model import JointContinuousVariable, CGParameter
import collections
import numpy as np
from pyltm.model.parameter import cgparameter
from pyltm.model.variable.discrete_variable import DiscreteVariable
from pyltm.model.parameter.cptparameter import CPTParameter
from pyltm.util.utils import logsumexp

class MixedCliqueSufficientStatistics(SufficientStatistics):
    '''
    classdocs
    '''


    def __init__(self, node, batch_size):
        '''
        node: Clique
        '''
        jointVariables = node.jointVariable
        discreteVariable = node.discreteVariable
        if isinstance(jointVariables, JointContinuousVariable):
            jointVariables = list(jointVariables.variables)
        elif isinstance(jointVariables, collections.Iterable):
            jointVariables = list(jointVariables)
        assert isinstance(jointVariables, list)
        self._continuousVariables = jointVariables
        self._discreteVariable = discreteVariable
        
        self.resetParameters(node.potential, batch_size)
        
    def resetParameters(self, cliquepotential, batch_size):
        cardinality = 1 if self._discreteVariable is None else self._discreteVariable.getCardinality()
        self.size = cardinality
        logp = cliquepotential.logp.copy()           # (K, )
        logconstant = logsumexp(logp)
        self.p = np.exp(logp - logconstant) # normalize
        self.mu = cliquepotential.mu.copy()         # (K, D)
        self.covar = cliquepotential.covar.copy()   # (K, D, D)
        # self.normalize()
        self.p = self.p * batch_size   # sufficient counts
        for i in range(cardinality):
            # sufficient sum_square
            self.covar[i] = (self.covar[i] + np.outer(self.mu[i], self.mu[i])) * self.p[i]
            # sufficient sum 
            self.mu[i] = self.mu[i] * self.p[i]
            
        
    def normalize(self, constant=None):
        if constant is None:
            constant = np.sum(self.p)
        self.p /= constant
        return constant
    
    def reset(self):
        self.p[:] = 0
        self.mu[:] = 0
        self.covar[:] = 0
        
    def add(self, potential):
        '''potential: batched cliquepotential'''
        batch_size = potential.logp.shape[0]
        # maybe normalize it in case hasn't been normalized
        logp = potential.logp - logsumexp(potential.logp, axis=1, keepdims=True)
        for i in range(potential.size):
            weight = np.expand_dims(np.exp(logp[:, i]), axis=1)  # (N, 1)
            self.p[i] += np.sum(weight)
            self.mu[i] += np.sum(potential.mu[:, i, :] * weight, axis=0)  # (N, D) x (N, 1)
            self.covar[i] += np.sum(np.concatenate([np.expand_dims(np.outer(potential.mu[j, i, :], potential.mu[j, i, :]) * weight[j], axis=0)
                              for j in range(batch_size)], axis=0), axis=0)
            
    def update(self, batchStatistics, learning_rate):
        assert(self.size==batchStatistics.size)
        self.p[:] = self.p + learning_rate * (batchStatistics.p - self.p)
        self.mu[:] = self.mu + learning_rate * (batchStatistics.mu - self.mu)
        self.covar[:] = self.covar + learning_rate * (batchStatistics.covar - self.covar)
        
    def computePotential(self, variable, parent):
        if isinstance(variable, JointContinuousVariable):
            parameters = [None]*self.size
            for i in range(self.size):
                parameters[i] = CGParameter(1, len(self.mu[i]), self.computeMean(self.p[i], self.mu[i]),
                                            self.computeCovariance(self.p[i], self.mu[i], self.covar[i]))
            return parameters
        elif isinstance(variable, DiscreteVariable):
            # only possibility is that variable is root
            parameter = CPTParameter(self.size)
            parameter.prob[:] = self.p
            parameter.normalize()
            return parameter
                
    def computeMean(self, p, mu):
        if p == 0:
            return np.zeros_like(mu)
        return mu / p
                
    def computeCovariance(self, p, mu, covar):
        if p==0:
            return np.ones_like(covar)
        mu = self.computeMean(p, mu)
        return covar / p - np.outer(mu, mu)
    