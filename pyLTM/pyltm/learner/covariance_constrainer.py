'''
Created on 22 Sep 2018

@author: Bryan
'''
import numpy as np
from pyltm.model.potential.cgpotential import CGPotential

class CovarianceConstrainer(object):
    '''
    classdocs
    '''


    def __init__(self, lower_bound=0.01):
        '''
        Constructor
        '''
        self.lower_bound = lower_bound
        
    def adjust(self, cgpotential):
        assert(isinstance(cgpotential, CGPotential))
        for i in range(cgpotential.size):
            parameter = cgpotential.get(i)
            try:
                eigvalues, eigvectors = np.linalg.eigh(parameter.covar)
            except:
                raise Exception("improper covariance matrix")
            
            changed = np.any(eigvalues<self.lower_bound)
            if changed:
                eigvalues = np.maximum(eigvalues, self.lower_bound)
                L = np.diag(eigvalues)
                parameter.covar[:] = eigvectors.dot(L).dot(eigvectors.T)
            