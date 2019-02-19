'''
Created on 20 Apr 2018

@author: Bryan
'''
import sys
from lark import Lark, Transformer

bif_grammar = r"""
    model: (network | variable | root_prob | nonroot_prob)+ 

    name : STRING
    variable_name : STRING
    type : CONTINUOUS | DISCRETE
    num_states : NUMBER
    state : STRING
    child : variable_name
    parent : variable_name
    table : "table" (NUMBER)+
    state_prob : "(" state ")" (NUMBER)+ ";"
    
    network : "network" name "{" "}"
    variable : "variable" variable_name "{" "type" type ["[" num_states "]"] ["{" (state)+ "}"] ";" "}"
    root_prob : "probability" "(" child ")" "{" (table ";")+"}" 
    nonroot_prob : "probability" "(" child "|" parent")" "{" (state_prob)+"}" 

    CONTINUOUS : "continuous"
    DISCRETE : "discrete"
    
    %import common.ESCAPED_STRING -> STRING
    %import common.SIGNED_NUMBER -> NUMBER
    %import common.WS
    %ignore WS
    """
    
bif_parser = Lark(bif_grammar, start='model', lexer='standard')
tree = bif_parser.parse(open("continuoustoy.bif").read())

# print(tree.pretty())
print(tree)

def read_row(t):
    row = []
    for ins in t:
        row.append(float(ins))
    return row
    
def read_tree(t):
    if t.data == "network":
        name = t.children[0].children[0].value
        print("network : ", name)
    if t.data == "variable":
        name = t.children[0].children[0].value
        type = t.children[1].children[0].value

        print("variable : %s, type: %s" % (name, type))
        if type == "discrete":
            num_states = t.children[2].children[0].value
            states = []
            for ins in t.children[3:]:
                states.append(ins.children[0].value)
            print("%d states: %s" % (int(num_states), " ".join(states)))
    if t.data == "root_prob":
        varname = t.children[0].children[0].children[0].value
        table = t.children[1].children
        prob = read_row(table)
        print("probability: ", varname)
        print("%s" % (" ".join([str(x) for x in prob])))
    if t.data == "nonroot_prob":
        child = t.children[0].children[0].children[0].value
        parent = t.children[1].children[0].children[0].value
        print("probability (%s | %s)" % (child, parent))
        prob = []
        for ins in t.children[2:]:
            state_prob = read_row(ins.children[1:])
            prob.append(state_prob)
        print("%s" % (" ".join([str(x) for x in prob])))
        
for inst in tree.children:
    read_tree(inst)