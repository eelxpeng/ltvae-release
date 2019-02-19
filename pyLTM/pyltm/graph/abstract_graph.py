'''
Created on 10 Feb 2018

@author: Bryan
'''
from abc import ABCMeta, abstractmethod
from .abstract_node import AbstractNode

class AbstractGraph(metaclass=ABCMeta):
    '''
    Abstract class for all graphs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self._nodes = list() # node list
        self._edges = list() # edge list
        self._names = dict() # HashMap: str : node
    
    @abstractmethod
    def addNode(self, name):
        '''
        Add a node to the graph with specified name
        '''
        pass
    @abstractmethod
    def addEdge(self, head, tail):
        pass
        
    def containsEdge(self, edge):
        return edge.getHead().getGraph() is self
    
    def containsNode(self, node):
        if isinstance(node, AbstractNode):
            return node.getGraph() is self
        elif isinstance(node, str):
            return node in self._names
        else:
            raise ValueError("Argument to Abstractgraph.containsNode error!")
    
    def removeNode(self, node):
        assert self.containsNode(node)
        edges = node.getEdges()
        for edge in edges:
            self.removeEdge(edge)
        self._nodes.remove(node)
        self._names.pop(node.name)
        
    @abstractmethod
    def removeEdge(self, edge):
        pass
        
    @property
    def edges(self):
        return self._edges
    
    @property
    def nodes(self):
        return self._nodes
    
    @property
    def names(self):
        return self._names.keys()
    
    def getNode(self, name):
        return self._names.get(name)
    
    def getNumberOfNodes(self):
        return len(self._nodes)
    
    def getNumberOfEdges(self):
        return len(self._edges)
    
    def __str__(self):
        toStr = "" + self.__class__.__name__ +"{\n"
        toStr += "number of nodes: " + str(self.getNumberOfNodes()) +"\n"
        toStr += "nodes = {\n"
        for node in self.nodes:
            toStr += str(node) + " "
        toStr += "\n}\n"
        toStr += "number of edges: " + str(self.getNumberOfEdges()) + "\n"
        toStr += "edges = {\n"
        for edge in self.edges:
            toStr += str(edge) + " "
        toStr += "\n}\n"
        toStr += "}\n"
        return toStr
    