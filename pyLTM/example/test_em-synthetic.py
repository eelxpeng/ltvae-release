'''
Created on 12 Sep 2018

@author: Bryan
'''
import sys
sys.path.append("..")

import numpy as np
import math
import time
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from pyltm.io import BifParser
from pyltm.learner import FullEM

if __name__ == '__main__':
    modelfile = "synthetic.bif"
    varNames = ["z1", "z2", "z3", "z4"]
    
    bifparser = BifParser()
    net = bifparser.parse(modelfile)
    print("Before EM")
    print(net)
    
    data = np.load("synthetic-2d-v2.npz")
    X = data["z"]
    
    learning_rate = 0.01
    batch_size = 100
    num_epochs = 50
    estimator = FullEM(net)
    
    start_time = time.time()
    estimator.fit(X, varNames, learning_rate, batch_size, num_epochs, initialize=True)
    elapsed_time = time.time() - start_time
    print("Elapsed time", elapsed_time)
    print("After EM")
    print(net)
    
    
