'''
Created on 10 Feb 2018

@author: Bryan
'''
from .abstract_graph import AbstractGraph
from .edge import Edge
from .directed_node import DirectedNode

class DirectedAcyclicGraph(AbstractGraph):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        super().__init__()
        
    def addEdge(self, head, tail):
        assert self.containsNode(head) and self.containsNode(tail)
        assert head is not tail
        assert not head.hasNeighbor(tail)
        # implement this check later
        # assert not self.containPath(head, tail)
        
        edge = Edge(head, tail)
        self._edges.append(edge)
        
        # attach the edge to both ends for general graph
        head.attachEdge(edge)
        tail.attachEdge(edge)
        
        # attach parent and children ends for directed graph
        head.attachInEdge(edge)
        tail.attachOutEdge(edge)
        
        return edge
    
    def addNode(self, name):
        name = name.strip()
        assert len(name)>0
        assert not self.containsNode(name)
        node = DirectedNode(self, name)
        self._nodes.append(node)
        self._names[name] = node
        return node
    
    def removeEdge(self, edge, vertex2=None):
        if isinstance(edge, Edge):
            assert self.containsEdge(edge)
            self._edges.remove(edge)
            # detach edge from both ends
            edge.head.detachEdge(edge)
            edge.tail.detachEdge(edge)
            edge.head.detachInEdge(edge)
            edge.tail.detachOutEdge(edge)
        
        elif isinstance(edge, DirectedNode) and isinstance(vertex2, DirectedNode):
            self.removeEdge(edge.getEdge(vertex2))
            
            
        
        
        
        