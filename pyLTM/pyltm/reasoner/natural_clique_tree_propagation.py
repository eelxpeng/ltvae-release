'''
Created on 13 Feb 2018

@author: Bryan
'''
from .natural_clique_tree import NaturalCliqueTree
from .clique import Clique, MixedClique
from ..model import Gltm
from .evidence import Evidence
import math
import numpy as np

class NaturalCliqueTreePropagation(object):
    '''
    natural clique tree propagation for inference
    '''


    def __init__(self, structure, focus=None):
        '''
        structure: Gltm
        focus: Focus
        '''
        self._model = structure
        self._tree = NaturalCliqueTree(structure)
        self._evidences = Evidence()
        self._loglikelihood = 0.
        self._evidenceAbsorbed = False
        self._messagesPassed = 0
        self._focusSpecified = False
        if focus is not None:
            self._focusSpecified = True
            self._tree.setFocus(focus)
            
    def cliqueTree(self):
        return self._tree
    
    def useModel(self, model):
        assert isinstance(model, Gltm)
        
    def propagate(self):
        '''
        Perform propagation and returns the likelihood
        '''
        self._messagesPassed = 0
        
        # initialization
        self.initializePotentials()
        self.absorbEvidence()
        
        if self._focusSpecified:
            self._evidenceAbsorbed = True
            
        self.collectMessage(self._tree.pivot)
        self.distributeMessage(self._tree.pivot)
        
        # keep the potential of pivot under proper normalization
        self._tree.pivot.normalize()
        self._loglikelihood = self._tree.pivot.logNormalization()
        
        self.setSeparatorPotentials()
        
        self.release()
        
        if np.isinf(self._loglikelihood).any() or np.isnan(self._loglikelihood).any():
            raise ValueError("ImpossibleEvidenceException")
        
    def initializePotentials(self):
        for node in self._tree.nodes:
            node.reset()
            
        for node in self._model.nodes:
            clique = self._tree.getClique(node.variable)
            #since tree use child node as the key, only P(x1|y1) will be combined into C(x1,y1), not p(y1)
            # unless it is the pivot. The root P(y1) will be combined to pivot clique
            clique.combine(node.potential)
            
    def absorbEvidence(self):
        '''
        absorb continuous evidence one singular variable by one singular variable
        should have a better way
        '''
#         for variable, value in self._evidences.entrySet():
#             clique = self._tree.getClique(variable)
#             
#             # ignore variable not contained in this model
#             if clique is None:
#                 continue
#             
#             if self._evidenceAbsorbed and not clique.focus():
#                 continue
#             
#             clique.absorbEvidence(variable, value)
        for clique in self._tree.cliques:
            if isinstance(clique, MixedClique):
                # list of singular continuous variables
                continuousVariables = clique.jointVariable.variables
                values = self._evidences.getValues(continuousVariables)
                clique.absorbEvidence(continuousVariables, values)
    
    def collectMessage(self, sink, separator=None, source=None):
        class CollectVisitor(Clique.NeighborVisitor):
            def visit(self, separator1, neighbor):
                self._ctp.collectMessage(sink, separator1, neighbor)
                
        class CollectVisitorRecursive(Clique.NeighborVisitor):
            def visit(self, separator1, neighbor):
                self._ctp.collectMessage(source, separator1, neighbor)
        
        if separator is None:
            sink.visitNeighbors(CollectVisitor(self, None))
        else:
            if separator.getMessage(source) is None:
                source.visitNeighbors(CollectVisitorRecursive(self, separator))
            
            self.sendMessage(source, separator, sink, False)
        
    def distributeMessage(self, source, separator=None, sink=None):
        class DistributeVisitor(Clique.NeighborVisitor):
            def visit(self, separator1, neighbor):
                self._ctp.distributeMessage(source, separator1, neighbor)
                
        class DistributeVisitorRecursive(Clique.NeighborVisitor):
            def visit(self, separator1, neighbor):
                self._ctp.distributeMessage(sink, separator1, neighbor)
        
        if separator is None:
            source.visitNeighbors(DistributeVisitor(self, None))
        else:
            if not sink.focus:
                return
            
            self.sendMessage(source, separator, sink, True)
            sink.visitNeighbors(DistributeVisitorRecursive(self, separator))
            
    def sendMessage(self, source, separator, sink, distributing):
        '''
        clique tree propagation has two pass: collecting and distributing
        When collecting, separator should have no sink message
        When distributing, separator should already have sink message
            In such case, the source message should divide the sink message 
        '''
        sourceMessage = separator.getMessage(source)
        if sourceMessage is None:
            sourceMessage = source.computeMessage(separator)
            separator.putMessage(source, sourceMessage)
            
        if distributing:
            sinkMessage = separator.getMessage(sink)
            # assert sinkMessage is not None
            if sinkMessage is not None:
                sourceMessage = sourceMessage.clone()
                sourceMessage.divide(sinkMessage)
                
        sink.combine(sourceMessage)
        self._messagesPassed += 1
        
    def setSeparatorPotentials(self):
        for sep in self._tree.separators():
            sep.setPotential()
            
    def release(self, force=False):
        for sep in self._tree.separators():
            sep.release(force or not self._focusSpecified)
            
    @property
    def model(self):
        return self._model
    @property
    def evidences(self):
        return self._evidences
    
    def use(self, evidence):
        if evidence is None:
            evidence = Evidence()
        self._evidences = evidence
        
    @property
    def loglikelihood(self):
        return self._loglikelihood
    @property
    def messagePassed(self):
        return self._messagesPassed
    
    def getMarginal(self, variable):
        return self._tree.getClique(variable).potential.marginalize(variable);

    
    
    
    
        
            
            
      
        
    
        
    