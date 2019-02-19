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
    n_samples = 1000
    n_centers = 2
    blobs = datasets.make_blobs(n_samples=n_samples, centers=2, random_state=8)
    X, y = blobs
#     gmm = mixture.GaussianMixture(n_components=n_centers)
#     gmm.fit(X)
#     print(gmm.weights_)
#     print(gmm.means_)
#     print(gmm.covariances_)
#     y_pred = gmm.predict(X)
#     colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                              '#f781bf', '#a65628', '#984ea3',
#                                              '#999999', '#e41a1c', '#dede00']),
#                                       int(max(y_pred) + 1))))
#     plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
#     plt.show()

    modelfile = "blob2cluster.bif"
    varNames = ["x","y"]
    bifparser = BifParser()
    net = bifparser.parse(modelfile)
    print("Before EM")
    print(net)
    
    learning_rate = 0.01
    batch_size = 100
    num_epochs = 10
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
    
    # set up evidence
    datacase = ContinuousDatacase.create(varNames)
    datacase.synchronize(net)
    ctp = NaturalCliqueTreePropagation(net)
    
    targetVar = None
    for v in net._variables.keys():
        if v.name == "z":
            targetVar = v
            break
    print(targetVar)
    y_pred = []
    
    datacase.putValues(X)
    evidence = datacase.getEvidence()
        
    ctp.use(evidence)
    ctp.propagate()
    latent = np.exp(ctp.getMarginal(targetVar).logprob)
    y_pred = np.argmax(latent, axis=1)
    
    print(y_pred)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                        '#f781bf', '#a65628', '#984ea3',
                                        '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    plt.show()
