'''
Created on 10 Feb 2018

@author: Bryan
'''
from .abstract_node import AbstractNode

class DirectedNode(AbstractNode):
    '''
    classdocs
    '''
    def __init__(self, graph, name):
        super().__init__(graph, name)
        self._parents = dict()   # Dict: DirectedNode: Edge
        self._children = dict()  # Dict: DirectedNode: Edge

    def attachInEdge(self, edge):
        self._parents[edge.tail] = edge
    
    def attachOutEdge(self, edge):
        self._children[edge.head] = edge
        
    def detachInEdge(self, edge):
        self._parents.pop(edge.tail, None)
        
    def detachOutEdge(self, edge):
        self._children.pop(edge.head, None)
        
    @property
    def children(self):
        return list(self._children.keys())
    
    @property
    def parents(self):
        return self._parents
    
    def isRoot(self):
        return len(self._parents) == 0
    
    def isLeaf(self):
        return len(self._children)==0
    
    def getParent(self):
        if self.isRoot():
            return None
        return list(self._parents.keys())[0]
    
    
        
        
        