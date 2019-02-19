'''
Created on 12 Sep 2018

@author: Bryan
'''
import numpy as np
from .sufficient_statistics import SufficientStatistics
from pyltm.model import CPTPotential

class DiscreteCliqueSufficientStatistics(SufficientStatistics):
    '''
    classdocs
    '''


    def __init__(self, node, batch_size):
        '''
        Constructor
        '''
        self._variables = node.potential._variables
        node.potential.normalize()
        self.prob = np.exp(node.potential.logprob.copy()) * batch_size
        
    def reset(self):
        self.prob[:] = 0
        
    def add(self, potential):
        self.prob[:] += np.sum(np.exp(potential.logprob), axis=0)
        
    def update(self, statistics, learning_rate):
        self.prob[:] = self.prob + learning_rate * (statistics.prob - self.prob)
        
    def computePotential(self, variable, parent):
        cptpotential = CPTPotential(self._variables)
        cptpotential.parameter.prob[:] = self.prob
        # sum out any irrelevant variables
        for v in self._variables:
            if v!=variable and (parent is None or v!=parent):
                cptpotential = cptpotential.sumOut(v)
        cptpotential.normalizeOver(variable)
        return cptpotential.parameter