'''
Created on 12 Feb 2018

@author: Bryan
'''
from .clique import Clique
from ..clique_potential import CliquePotential, MixedCliquePotential
from ..message import Message
from pyltm.model.potential.cgpotential import CGPotential
from pyltm.reasoner.clique_potential.discrete_clique_potential import DiscreteCliquePotential
from pyltm.model.potential.cptpotential import CPTPotential

class MixedClique(Clique):
    '''
    classdocs
    '''


    def __init__(self, tree, name, jointVar, discreteVar):
        '''
        Constructor
        '''
        super().__init__(tree, name)
        self._jointVariable = jointVar
        self._discreteVariable = discreteVar
        
    @property
    def discreteVariable(self):
        return self._discreteVariable
    
    @property
    def jointVariable(self):
        return self._jointVariable
    
    @property
    def discreteVariables(self):
        return [self._discreteVariable]
    
    @property
    def potential(self):
        return self._potential
    
    def logNormalization(self):
        return self._potential.logNormalization
    
    def contains(self, variable):
        return self._discreteVariable is variable or self._jointVariable is variable or \
            variable in self._jointVariable.variables
            
    def assign(self, potential):
        """
        potential: CGPotential
        """
        self._potential = MixedCliquePotential(potential)
        
    def absorbEvidence(self, variable, value):
        self._potential.absorbEvidence(variable, value)
        
    def computeMessage(self, separator, multiplier=None, retainingVariables=None):
        """
        Should only contain discrete variable of the separator
        Always retain the discrete variable and marginalize out continuous
        """
        discreteCliquePotential = self._potential.marginalize(separator.variable)
        message = Message(discreteCliquePotential)
        return message.times(multiplier) if multiplier is not None else message
    
    def reset(self):
        self._potential = None
        
    def combine(self, other, logNormalization=0):
        """
        other: Potential
        """
        if isinstance(other, Message):
            self.combine(other.function())
            return
        elif isinstance(other, CGPotential):
            if self._potential is None:
                self._potential = MixedCliquePotential(other.clone(), logNormalization)
            else:
                other = MixedCliquePotential(other, logNormalization)
                self._potential.combine(other)
        elif isinstance(other, CPTPotential):
            # other is CPTPotential
            other = other.function()
            if self._potential is None:
                cgpotential = CGPotential(self._jointVariable, self._discreteVariable)
                self._potential = MixedCliquePotential(cgpotential, logNormalization)
            other = DiscreteCliquePotential(other, logNormalization)
            self._potential.multiply(other)
        elif isinstance(other, DiscreteCliquePotential):
            self._potential.multiply(other)
        else:
            raise Exception("Invalid potential type!")
                
        if self.pivot:
            self.normalize()
        
    def variables(self):
        variables = self._jointVariable.variables() + [self._discreteVariable]
        return variables
        
    def __str__(self):
        toStr = "" + self.__class__.__name__ + ": " 
        toStr += " ".join([v.name for v in self._jointVariable.variables])
        toStr += " " + self._discreteVariable.name + "\n"
        toStr += "neighbors={ " + " ".join([n.name for n in self.getNeighbors()]) + " }\n"
        
        if self.potential is not None:
            toStr += str(self.potential)
            toStr += "\n"
        else:
            toStr += "potential: None\n"
        toStr += "}\n"
            
        return toStr