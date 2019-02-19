'''
Created on 10 Feb 2018

@author: Bryan
'''
from abc import ABCMeta, abstractmethod

class AbstractNode(metaclass=ABCMeta):
    '''
    classdocs
    '''
    def __init__(self, graph, name):
        '''
        Constructor
        '''
        self._graph = graph
        self._name = name
        self._neighbors = dict() # HashMap: AbstractNode: Edge
        
    def attachEdge(self, edge):
        opposite = edge.getOpposite(self)
        self._neighbors[opposite] = edge
        
    def detachEdge(self, edge):
        self._neighbors.pop(edge.getOpposite(self), None)
        
    def dispose(self):
        self._graph = None
        
    def getDegree(self):
        return len(self._neighbors)
    
    def getEdges(self):
        return self._neighbors.values()
    
    def getGraph(self):
        return self._graph
    
    def getEdge(self, node):
        assert node in self._neighbors
        return self._neighbors.get(node)
    
    @property
    def name(self):
        return self._name
    
    def __str__(self):
        return self.name
    
    def getNeighbors(self):
        return self._neighbors.keys()
    
    def hasNeighbor(self, node):
        return node in self._neighbors
    
    def setName(self, name):
        self._name = name
    
    