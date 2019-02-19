'''
Created on 10 Feb 2018

@author: Bryan
'''

from pyltm.graph.directed_acyclic_graph import DirectedAcyclicGraph
"""
Testing of directed acyclic graph
"""
# graph y -> z -> x
dag = DirectedAcyclicGraph()
y = dag.addNode("y")
z = dag.addNode("z")
x = dag.addNode("x")
dag.addEdge(z, y)
dag.addEdge(x, z)
print(str(dag))
