'''
Created on 14 Feb 2018

@author: Bryan
'''
import sys
sys.path.append("..")

from pyltm.model import Gltm
from pyltm.model import DiscreteVariable, SingularContinuousVariable
from pyltm.reasoner import NaturalCliqueTreePropagation, Evidence
import numpy as np

if __name__ == '__main__':
    # test continuous BayesNet
    net = Gltm("continuousToy")
    y = DiscreteVariable("y", 2)
    z = DiscreteVariable("z", 2)
    x = SingularContinuousVariable("x")
    node_y = net.addNode(y)
    node_z = net.addNode(z)
    node_x = net.addNode(x)
    net.addEdge(node_z, node_y)
    net.addEdge(node_x, node_z)
    
    mus = np.array([[0.], [4.]])
    covars = np.array([[[1.]], [[4.]]])
    node_x.potential.setEntries(mus, covars)
    print(str(net))
    
    # set up evidence
    evidence = Evidence()
    evidence.add(x, 0)
    
    ctp = NaturalCliqueTreePropagation(net)
    print(ctp._tree)
    ctp.use(evidence)
    ctp.propagate()
    loglikelihood = ctp.loglikelihood
    print("Loglikelihood: ", loglikelihood)