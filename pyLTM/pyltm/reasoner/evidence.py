'''
Created on 14 Feb 2018

@author: Bryan
'''
import math
import numpy as np
import collections

class Evidence(object):
    '''
    classdocs
    '''


    def __init__(self, other=None):
        '''
        Constructor
        '''
        if other is None:
            self._entries = dict()  # Variable: double
        else:
            self._entries = other.entries.copy()
            
    def project(self, variables):
        '''list/set of Variable'''
        projected = Evidence()
        for var in self._entries:
            if var in variables:
                projected.add(var, self._entries[var])
        
        return projected
    
    def add(self, variable, value):
        if isinstance(variable, collections.Iterable):
            for i in range(len(variable)):
                self.add(variable[i], value[:, i])
        else:
            self._entries[variable] = value
    
    def getValues(self, variables):
        values = np.hstack([np.expand_dims(self._entries[v], axis=1) for v in variables])
        return values
    
    def clear(self):
        self._entries.clear()
        
    def clone(self):
        return Evidence(self)
    
    def entrySet(self):
        return self._entries
    
    def get(self, variable):
        return self._entries[variable]
    
    def __str__(self):
        return str(self._entries)
            
    