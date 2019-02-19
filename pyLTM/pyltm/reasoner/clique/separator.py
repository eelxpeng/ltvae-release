'''
Created on 13 Feb 2018

@author: Bryan
'''
from .clique_tree_node import CliqueTreeNode
from ..message import Message

class Separator(CliqueTreeNode):
    '''
    classdocs
    '''

    def __init__(self, tree, variable):
        '''
        Constructor
        '''
        super().__init__(tree, variable.name)
        self._variable = variable
        self._messages = dict()   # Dict: Clique: Message
        self._lastMessage = None
        self._potential = None
        
    def reset(self):
        '''do not clear the messages'''
        self._potential = None 
    
    @property    
    def variable(self):
        return self._variable
    
    @property
    def potential(self):
        return self._potential
    
    def putMessage(self, clique, message):
        assert isinstance(message, Message)
        self._messages[clique] = message
        self._lastMessage = message
        
    def getMessage(self, clique):
        return None if clique not in self._messages else self._messages[clique]
        
    def setPotential(self):
        '''
        set the potential to the last message
        '''
        if self._lastMessage is not None:
            self._potential = self._lastMessage.potential.clone()
    
    @property
    def variables(self):
        return [self._variable]
    
    @property
    def discreteVariables(self):
        return [self._variable]
    
    def release(self, isall=True):
        '''release all messages, or only those in focus'''
        if self._messages is None:
            return
        if isall:
            self._messages.clear()
        else:
            for clique in self._messages:
                if clique.focus():
                    self._messages.pop(clique)
    
    def withinFocusBoundary(self):
        '''
        check if this separator is within boundary of the focus subtree
        '''
        for clique in self.getNeighbors():
            if clique.focus():
                return True
        return False
    
    def createMessageMemento(self):
        return Separator.MessageMemento(self._messages)
    
    def setMessageMemento(self, memento):
        if memento is None or len(memento)==0:
            return
        if self._messages is None:
            self._messages = dict()
        else:
            self._messages.clear()
        memento.putMessagesInto(self._messages)
        
    class MessageMemento:
        def __init__(self, messages):
            '''
            messages: dict Clique : Message
            '''
            if messages is None or len(messages)==0:
                self._messages = None
            else:
                # _messages also a dict
                self._messages = messages.copy()
                
        def putMessageInto(self, messages):
            for clique in self._messages:
                messages[clique] = self._messages[clique]
                
        def isEmpty(self):
            return self._messages is None or len(self._messages)==0
                
    
    