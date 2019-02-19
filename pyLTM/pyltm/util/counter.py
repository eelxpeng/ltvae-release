'''
Created on 11 Feb 2018

@author: Bryan
'''

class Counter(object):
    '''
    classdocs
    '''
    class Instance():
        def __init__(self, index, name):
            self.index = index
            self.name = name

    def __init__(self, prefix):
        '''
        Constructor
        '''
        self.current = 0
        self.prefix = prefix
        
    def next(self):
        instance = Counter.Instance(self.current, self.createName())
        self.current += 1
        return instance.index, instance.name
    
    def nextIndex(self):
        self.current += 1
        return self.current
    
    def encounterName(self, name):
        pass
    
    def createName(self):
        return self.prefix+str(self.current)
    