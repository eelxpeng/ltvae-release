'''
Created on 10 Feb 2018

@author: Bryan
'''

class Edge(object):
    '''
    classdocs
    '''

    def __init__(self, head, tail):
        '''
        Constructor
        '''
        self._head = head
        self._tail = tail
        
    @property
    def head(self):
        return self._head
    @property
    def tail(self):
        return self._tail
    
    def getOpposite(self, end):
        assert end is self._head or end is self._tail
        return self._tail if end is self._head else self._head
    
    def __str__(self):
        return "(" + self._tail.name + " -> " + self._head.name + ")"
    
        