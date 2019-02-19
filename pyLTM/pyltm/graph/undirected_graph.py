'''
Created on 12 Feb 2018

@author: Bryan
'''
from .abstract_graph import AbstractGraph
from .edge import Edge
from .undirected_node import UndirectedNode

class UndirectedGraph(AbstractGraph):
    '''
    classdocs
    '''

    def addEdge(self, head, tail):
        assert self.containsNode(head) and self.containsNode(tail)
        assert head is not tail
        assert not head.hasNeighbor(tail)
        edge = Edge(head, tail)
        self._edges.append(edge)
        head.attachEdge(edge)
        tail.attachEdge(edge)
        return edge
    
    def addNode(self, node):
        if isinstance(node, str):
            name = node.strip()
            assert len(name)>0
            assert not self.containsNode(name)
            node = UndirectedNode(self, name)
        self._nodes.append(node)
        self._names[node.name] = node
        return node
    
    def removeEdge(self, edge):
        assert self.containsEdge(edge)
        self._edges.remove(edge)
        edge.head.detachEdge(edge)
        edge.tail.detachEdge(edge)
        
        