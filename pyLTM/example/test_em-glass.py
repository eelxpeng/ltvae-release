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
from pyltm.learner import EMFramework
from pyltm.data import ContinuousDatacase
from pyltm.reasoner import NaturalCliqueTreePropagation, Evidence

if __name__ == '__main__':
    modelfile = "glass.bif"
    varNames = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]
    X = np.array([[1.51793,12.79,3.5,1.12,73.03,0.64,8.77,0,0],
                     [1.51643,12.16,3.52,1.35,72.89,0.57,8.53,0,0]])
    
    bifparser = BifParser()
    net = bifparser.parse(modelfile)
    print("Before EM")
    print(net)
    
    learning_rate = 0.01
    batch_size = 2
    num_epochs = 1
    em = EMFramework(net, batch_size)
    
    start_time = time.time()
    for epoch in range(num_epochs):
        num_iters = int(math.floor(len(X))/batch_size)
        for i in range(num_iters):
            batchX = X[i*batch_size:min((i+1)*batch_size, len(X))]
            em.stepwise_em_step(batchX, varNames, learning_rate)
    elapsed_time = time.time() - start_time
    print("Elapsed time", elapsed_time)
    print("After EM")
    print(net)
    
    
