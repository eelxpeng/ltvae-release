'''
Created on 12 Feb 2018

@author: Bryan
'''
from .clique import Clique
from ..message import Message
from pyltm.reasoner.clique_potential import CliquePotential, DiscreteCliquePotential
from pyltm.model import Potential, CPTPotential, CGPotential

class DiscreteClique(Clique):
    '''
    classdocs
    '''


    def __init__(self, tree, name, variables):
        '''
        variables: list of Variable
        '''
        super().__init__(tree, name)
        self._variables = list(variables)
        
    @property
    def variables(self):
        return self._variables
    @property
    def discreteVariables(self):
        return self._variables
    @property
    def potential(self):
        return self._potential
    
    def logNormalization(self):
        return self._potential.logNormalization
        
    def contains(self, variable):
        return variable in self._variables
    
    def computeMessage(self, separator, multiplier=None, retainingVariables=None):
        """
        multiplier: Message from upstream
        """
        cliquepotential = self._potential if multiplier is None else self._potential.times(multiplier.potential)
        for variable in self._potential.variables:
            if variable is separator.variable:
                continue
            if retainingVariables is not None and variable in retainingVariables:
                continue
            cliquepotential = cliquepotential.sumOut(variable)
        return Message(cliquepotential)
        
    def reset(self):
        self._potential = None
    
    def combine(self, other, logNormalization=0):
        """
        other: Potential
        """
        if isinstance(other, Message):
            self.combine(other.function())
            return
        if self._potential is None:
            cptpotential = other.clone()
            self._potential = DiscreteCliquePotential(cptpotential, logNormalization)
        elif isinstance(other, Potential):
            other = other.function()
            other = DiscreteCliquePotential(other, logNormalization)
            self._potential = self._potential.times(other)
        elif isinstance(other, CliquePotential):
            other = other.function()
            self._potential = self._potential.times(other)
        else:
            raise Exception("invalid potential type")
            
        if self.pivot:
            self.normalize()
             
            