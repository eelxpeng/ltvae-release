'''
Created on 13 Feb 2018

@author: Bryan
'''

class CliquePotential(object):
    '''
    classdocs
    '''


    def __init__(self, potential, constant=0):
        '''
        Constructor
        '''
        self.content = potential
        self.logNormalization = constant
        
    def __str__(self):
        return str(self.content)