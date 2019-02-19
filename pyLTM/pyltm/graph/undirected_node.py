'''
Created on 12 Feb 2018

@author: Bryan
'''
from .abstract_node import AbstractNode
class UndirectedNode(AbstractNode):
    '''
    classdocs
    '''
    def computeDifficiency(self):
        difficiency = 0
        for neighbor1 in self.getNeighbors():
            for neighbor2 in self.getNeighbors():
                if not neighbor1.hasNeighbor(neighbor2):
                    difficiency += 1
        difficiency /= 2
        return difficiency
    
    def __str__(self):
        toStr = "Undirected node {\n"
        toStr += "\tname: "+self.name + "\n"
        toStr += "\tdegree: " + str(self.getDegree()) + "\n"
        toStr += "\tneighbors = [ "
        toStr += " ".join(self.getNeighbors())
        toStr += " ]\n"
        toStr += "}\n"
        return toStr