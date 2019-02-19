'''
Created on 18 Sep 2018

@author: Bryan
'''
import pdb
import numpy as np
import scipy.stats as stats
from .clique_potential import CliquePotential
from .discrete_clique_potential import DiscreteCliquePotential
from pyltm.model.variable import Variable, JointContinuousVariable, DiscreteVariable
from pyltm.model.potential import CGPotential, CPTPotential
from pyltm.util import logsumexp

class MixedCliquePotential(CliquePotential):
    '''
    classdocs
    '''


    def __init__(self, potential, constant=0, eps=1e-100):
        '''
        potential: cgpotential
        self.p: (K)
        self.mu: (K, D)
        self.covar: (K, D, D)
        '''
        self._continuousVariables = potential._continuousVariables
        self._discreteVariable = potential._discreteVariable
        
        K = potential.size
        D = potential.dimension
        self.size = K
        self.dimension = D
        # self.logp = np.log(np.array([potential.get(i).p for i in range(potential.size)]))
        self.logp = np.log(np.maximum(np.array([potential.get(i).p for i in range(potential.size)]), eps))
        self.mu = np.vstack([potential.get(i).mu for i in range(potential.size)])
        self.covar = np.concatenate([np.expand_dims(potential.get(i).covar, axis=0) for i in range(potential.size)], axis=0)
        self.logNormalization = constant
        
    def combine(self, other):
        '''other: MixedCliquePotential'''
        assert(self.size == other.size);
        self.logp += other.logp
        self.mu[:] = other.mu
        self.covar[:] = other.covar
        self.logNormalization += other.logNormalization
        
    def isBatch(self):
        return len(self.logp.shape) == 2
    
    def multiply(self, otherDiscreteCliquePotential):
        # self.p *= otherDiscreteCliquePotential.prob
        self.logp += otherDiscreteCliquePotential.logprob
        self.logNormalization += otherDiscreteCliquePotential.logNormalization
        
    def normalize(self, constant=None):
        if constant is None:
            if self.isBatch():
                batch_size = self.logp.shape[0]
                logconstant = logsumexp(self.logp, axis=1, keepdims=True)
                lognormalizeConstant = np.reshape(logconstant, (batch_size, ))
            else:
                logconstant = logsumexp(self.logp)
                lognormalizeConstant = logconstant
        try:
            self.logp[:] = self.logp - logconstant
            self.logNormalization += lognormalizeConstant
        except:
            pdb.set_trace()
            raise Exception("error!")
        
    def absorbEvidence(self, variables, values):
        '''
        taking list of singular continous variables, and array of values
        values: (batch_size, dim)
        use multivariate normal distribution to get prob
        '''
        index = [variables.index(v) for v in self._continuousVariables]
        synced_values = values[:, index]
        num, _ = synced_values.shape
        batchp = np.zeros((num, self.size))
#         for i in range(self.size):
#             batchp[:, i] = self.p[i]*stats.multivariate_normal.pdf(synced_values, mean=self.mu[i], cov=self.covar[i])
        for i in range(self.size):
            # why previously it is self.p[i]*logpdf ??? does not make sense.
            batchp[:, i] = self.logp[i] + stats.multivariate_normal.logpdf(synced_values, mean=self.mu[i], cov=self.covar[i])
        maxlogP = np.max(batchp, axis=1, keepdims=True)
        self.logp = batchp - maxlogP
        self.logNormalization += np.squeeze(maxlogP, axis=1)
        self.mu = np.repeat(np.expand_dims(synced_values, axis=1), self.size, axis=1)
        self.covar = np.zeros((num, self.size, self.dimension, self.dimension))
        return maxlogP
    
    def marginalize(self, variable):
        """
        only support marginalization of discrete variable
        simply return the p
        """
        assert isinstance(variable, DiscreteVariable)
        return self.function()
        
    def function(self):
        '''
        return DiscreteCliquePotential of discrete variable
        '''
        cpt = DiscreteCliquePotential(CPTPotential(self._discreteVariable))
        cpt.logprob = self.logp.copy()
        cpt.logNormalization = self.logNormalization.copy()
        return cpt
    
    def clone(self):
        continuousVars = list(self._continuousVariables)
        discreteVar = self._discreteVariable
        cgpotential = CGPotential(continuousVars, discreteVar)
        potential = MixedCliquePotential(cgpotential)
        potential.logp = self.logp.copy()
        potential.mu = self.mu.copy()
        potential.covar = self.covar.copy()
        potential.logNormalization = self.logNormalization
        return potential

    
