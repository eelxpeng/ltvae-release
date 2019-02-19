'''
Created on 20 Apr 2018

@author: Bryan
'''
import sys
from lark import Lark
from pyltm.model import Gltm
from pyltm.model import DiscreteVariable, SingularContinuousVariable
import numpy as np
from pyltm.model.node import DiscreteBeliefNode, ContinuousBeliefNode
from pyltm.reasoner import NaturalCliqueTreePropagation, Evidence

bif_grammar = r"""
    model: (network | variable | root_prob | nonroot_prob)+ 
    
    name : STRING
    variable_name : STRING
    type : CONTINUOUS | DISCRETE
    num_states : NUMBER
    state : STRING
    child : variable_name ("," variable_name)*
    parent : (variable_name)+
    table : "table" (NUMBER)+
    state_prob : "(" state ")" (NUMBER)+ ";"
    
    network : "network" name "{" "}"
    variable : "variable" variable_name "{" "type" type ["[" num_states "]"] ["{" (state)+ "}"] ";" "}"
    root_prob : "probability" "(" child ")" "{" (table ";")+"}" 
    nonroot_prob : "probability" "(" child "|" parent")" "{" (state_prob)+"}" 

    CONTINUOUS : "continuous"
    DISCRETE : "discrete"
    
    COMMENT : "//" /(.)+/
    %ignore COMMENT
    
    %import common.ESCAPED_STRING -> STRING
    %import common.SIGNED_NUMBER -> NUMBER
    %import common.WS
    %ignore WS
    """

def read_row(t):
    row = []
    for ins in t:
        row.append(float(ins))
    return row

def remove_quote(s):
    s2 = s.strip('\"')
    return s2
        
class BifParser:
    def __init__(self):
        self.bif_parser = Lark(bif_grammar, start='model', lexer='standard')
        
    def parse(self, filename):
        with open(filename) as fid:
            tree = self.bif_parser.parse(fid.read())
        self.net = None
        for inst in tree.children:
            self.read_tree(inst)
        
        return self.net
            
    def read_tree(self, t):
        if t.data == "network":
            name = t.children[0].children[0].value
            name = remove_quote(name)
            self.net = Gltm(name)
        if t.data == "variable":
            name = t.children[0].children[0].value
            name = remove_quote(name)
            type = t.children[1].children[0].value

            if type == "discrete":
                num_states = int(t.children[2].children[0].value)
                variable = DiscreteVariable(name, num_states)
                self.net.addNode(variable)
                states = []
                for ins in t.children[3:]:
                    states.append(ins.children[0].value)

            if type == "continuous":
                variable = SingularContinuousVariable(name)
                self.net.addNode(variable)
        if t.data == "root_prob":
            varname = t.children[0].children[0].children[0].value
            varname = remove_quote(varname)
            node = self.net.getNode(varname)
            variable = node.variable
            
            table = t.children[1].children
            prob = np.array(read_row(table))
            node.potential.setCells([variable], prob)
            
        if t.data == "nonroot_prob":
            childins = t.children[0].children
            children = []
            for ins in childins:
                child = ins.children[0].value
                child = remove_quote(child)
                children.append(child)
            parent = t.children[1].children[0].children[0].value
            parent = remove_quote(parent)
            
            childnodes = [self.net.getNode(child) for child in children]
            # only one parent
            parentnode = self.net.getNode(parent)
            
            prob = []
            for ins in t.children[2:]:
                state_prob = read_row(ins.children[1:])
                prob.append(state_prob)
            prob = np.array(prob)
            if isinstance(childnodes[0], DiscreteBeliefNode):
                for childnode in childnodes:
                    self.net.addEdge(childnode, parentnode)
                    childnode.potential.setCells([parentnode.variable, childnode.variable], prob)
            if isinstance(childnodes[0], ContinuousBeliefNode):
                newnode = self.net.combine(True, childnodes)
                self.net.addEdge(newnode, parentnode)
                dim = len(newnode.variable.variables)
                mus = prob[:, :dim]
                rest = prob[:, dim:]
                covs = []
                for row in rest:
                    cov = row.reshape((1, dim, dim))
                    covs.append(cov)
                covs = np.concatenate(covs, axis=0)
                newnode.potential.setEntries(mus, covs)
        
if __name__=="__main__":
    bifparser = BifParser()
    net = bifparser.parse("continuoustoy.bif")
    
    # set up evidence
    evidence = Evidence()
    x = net.getNode("x").variable.variables[0]
    evidence.add(x, 0)
    
    ctp = NaturalCliqueTreePropagation(net)
    print(ctp._tree)
    ctp.use(evidence)
    ctp.propagate()
    loglikelihood = ctp.loglikelihood
    print("Loglikelihood: ", loglikelihood)
    