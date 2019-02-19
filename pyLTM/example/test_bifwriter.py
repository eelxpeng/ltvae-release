'''
Created on 14 Feb 2018

@author: Bryan
'''
import sys
sys.path.append("..")

from pyltm.model import Gltm
from pyltm.model import DiscreteVariable, SingularContinuousVariable
from pyltm.reasoner import NaturalCliqueTreePropagation, Evidence
from pyltm.io import BifParser, BifWriter
from pyltm.data import ContinuousDatacase
import numpy as np

if __name__ == '__main__':
    modelfile = "glass.bif"
    
    bifparser = BifParser()
    net = bifparser.parse(modelfile)
    print(net)
    
    bifwriter = BifWriter()
    bifwriter.write(net, "glass-write.bif")
    