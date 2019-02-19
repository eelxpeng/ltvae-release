'''
Created on 13 Feb 2018

@author: Bryan
'''
from ..graph import UndirectedGraph
from ..model import DiscreteVariable, DiscreteBeliefNode, ContinuousBeliefNode
from .clique import Clique, Separator, DiscreteClique, MixedClique

class NaturalCliqueTree(UndirectedGraph):
    '''
    maintain tree with cliques and separators
    cliques may be discrete clique or mixed clique
    '''

    def __init__(self, structure=None):
        '''
        _separator: map from variable to separator
        _cliques: map from varaible to a clique containing that variable
                - key variable is the child of an edge
                - a discrete variable may be mapped to a mixed clique  
        '''
        super().__init__()
        self._separators = dict() # Variable : Separator
        self._cliques = dict()  # Variable: Clique
        self._pivot = None      # clique holding the root
        
        if structure is not None:
            # construct the separators
            for node in structure.nodes:
                if node.getDegree()>1:
                    # only internal discrete nodes can have separator
                    # All leaf nodes are continuous
                    self.addSeparator(node)
                    
            # construct the cliques by adding parent and child pair
            for edge in structure.edges:
                childNode = edge.head
                parentNode = edge.tail
                # child variable as the key
                clique = self.addClique(parentNode, childNode)
                # connect the clique with separators
                # edge should have no direction, but...
                if parentNode.getDegree()>1:
                    self.addEdge(clique, self._separators[parentNode.variable])
                if childNode.getDegree()>1:
                    self.addEdge(self._separators[childNode.variable], clique)
                
            # if the model contains only one node, add that single node as a clique
            if len(structure.nodes)==1:
                raise Exception("Does not support only one node")
            
            # use the clique with the root variable as the pivot
            # find where to attach root P(Y), and attach to it
            rootNode = structure.getRoot()
            self.setPivot(None if rootNode.variable not in self._cliques else self._cliques[rootNode.variable])
            # but if root variable does not as key in _clique
            # then use one of its children as the pivot
            # the root variable is associated with this pivot
            if self.pivot is None:
                childNode = rootNode.children[0]
                self.setPivot(self._cliques[childNode.variable])
                self._cliques[rootNode.variable] = self.pivot
                
    @property
    def pivot(self):
        return self._pivot
    
    def addSeparator(self, node):
        assert isinstance(node, DiscreteBeliefNode)
        variable = node.variable
        separator = Separator(self, variable)
        self.addNode(separator)
        self._separators[variable] = separator
        return separator
    
    def getSeparator(self, variable):
        return self._separators[variable]
    
    def addClique(self, parent, child):
        clique = None
        if isinstance(child, DiscreteBeliefNode):
            clique = DiscreteClique(self, self.newCliqueName(),
                            [child.variable, parent.variable])
        elif isinstance(child, ContinuousBeliefNode):
            clique = MixedClique(self, self.newCliqueName(),
                            child.variable, parent.variable)
        self.addNode(clique)
        self._cliques[child.variable] = clique
        return clique
    
    def setPivot(self, clique):
        if self._pivot is not None:
            self._pivot.setPivot(False)
        self._pivot = clique
        if self._pivot is not None:
            self._pivot.setPivot(True)
            
    def setFocus(self, focus):
        assert focus is not None
        for clique in self._cliques.values():
            clique.setFocus(False)
        
        for var in self._cliques.keys():
            if focus.contains(var):
                self._cliques[var].setFocus(True)
        
        #set the pivot to the clique holding the first variable
        if focus.size>0:
            self.setPivot(self._cliques[focus.pivot()])
            
    def newCliqueName(self):
        return "Clique" + str(self.getNumberOfNodes())
    
    def getClique(self, variable):
        return self._cliques[variable]
    
    def separators(self):
        return self._separators.values()
    
    @property
    def cliques(self):
        # return self._cliques.values()
        return [node for node in self.nodes if isinstance(node, Clique)]
    
    '''
    Currently not implementing Subtree and findMinimalSubtree
    since they appears to not be the functionality of inference
    but structure learning
    '''
    
        