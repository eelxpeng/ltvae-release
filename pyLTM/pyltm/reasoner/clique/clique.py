'''
Created on 12 Feb 2018

@author: Bryan
'''
from .clique_tree_node import CliqueTreeNode
from ..message import Message
from abc import ABCMeta, abstractmethod
import math
class Clique(CliqueTreeNode):
    '''
    classdocs
    '''
    class NeighborVisitor(metaclass=ABCMeta):
        """
        visitor pattern. override visit method for various operations
        Constructs this visitor by specifying an origin of visit. 
        The origin separator is skipped when enumerating the neighbor separators in the visit.
        """
        def __init__(self, ctp, origin=None):
            self._ctp = ctp
            self._origin = origin
        
        @property
        def origin(self):
            return self._origin
        
        @abstractmethod
        def visit(self, separator, neighbor):
            pass

    def __init__(self, tree, name):
        '''
        Constructor
        '''
        super().__init__(tree, name)
        self._focus = True
        self._pivot = False
        self._potential = None
        
    @abstractmethod
    def contains(self, variable):
        pass
    
    def visitNeighbors(self, visitor):
        for separator in self.getNeighbors():
            if separator is visitor.origin:
                continue
            for neighbor in separator.getNeighbors():
                if neighbor is not self:
                    visitor.visit(separator, neighbor)
                    
    @abstractmethod
    def computeMessage(self, separator, multiplier=None, retainingVariables=None):
        """
        Computes the message for sending to the neighbor {@code separator}, by
        first multiplying the potential by the {@code multiplier}, and retains
        the given {@code retainingVariables} and does not marginalize out them.
        """
        pass
    
    @abstractmethod
    def combine(self, potential, logNormalization=0):
        if isinstance(potential, Message):
            potential = potential.potential
        pass
    
    def setFocus(self, focus=True):
        self.focus = focus
        
    @property
    def focus(self):
        return self._focus
    @property
    def pivot(self):
        return self._pivot
    
    def setPivot(self, pivot=True):
        self._pivot = pivot
        
    def isPivot(self):
        return self._pivot
    
    def normalize(self, constant=None):
        self.potential.normalize(constant)
