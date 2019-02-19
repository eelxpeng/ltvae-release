'''
Created on 12 Feb 2018

@author: Bryan
'''
from ...graph import UndirectedNode
from abc import abstractmethod, abstractproperty

class CliqueTreeNode(UndirectedNode):
    '''
    classdocs
    '''
    @abstractproperty
    def potential(self):
        pass
    @abstractproperty
    def variables(self):
        pass
    @abstractproperty
    def discreteVariables(self):
        pass
    
    def __str__(self):
        toStr = "" + self.__class__.__name__ + ": " + " ".join([v.name for v in self.variables]) + "\n"
        toStr += "neighbors={ " + " ".join([n.name for n in self.getNeighbors()]) + " }\n"
        
        if self.potential is not None:
            toStr += str(self.potential)
            toStr += "\n"
        else:
            toStr += "potential: None\n"
        toStr += "}\n"
            
        return toStr