'''
Created on 13 Feb 2018

@author: Bryan
'''
from .clique_potential import DiscreteCliquePotential

class Message(object):
    '''
    classdocs
    '''


    def __init__(self, potential):
        '''
        Constructor
        '''
        self.potential = potential
        
    def clone(self):
        return Message(self.potential.clone())
    
    def times(self, message):
        return Message(self.potential.times(message.potential))
    
    @staticmethod
    def computeProduct(messages):
        """
        messages: list of Message
        """
        raise Exception("computeProduct not implemented!")
#         cpt_list = [m.cptpotential for m in messages]
#         logProduct = sum([m.logNormalization for m in messages])
#         return Message(CPTPotential.computeProduct(cpt_list), logProduct)
    
    def divide(self, divider):
        '''divider: Message'''
        if isinstance(divider, DiscreteCliquePotential):
            self.potential.divide(divider)
        elif isinstance(divider, Message):
            self.potential.divide(divider.potential)
        
    def function(self):
        return self.potential
        