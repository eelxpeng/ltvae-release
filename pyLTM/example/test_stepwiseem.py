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
from pyltm.model import Gltm, DiscreteBeliefNode, ContinuousBeliefNode
from gaussian_mixture_stepwise import GaussianMixture

if __name__ == '__main__':
    n_samples = 1000
    n_centers = 2
    blobs = datasets.make_blobs(n_samples=n_samples, centers=2, random_state=8)
    X, y = blobs
    # gmm = mixture.GaussianMixture(n_components=n_centers)
    # gmm.fit(X)
    # print(gmm.weights_)
    # print(gmm.means_)
    # print(gmm.covariances_)
    # y_pred = gmm.predict(X)
    # colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
    #                                          '#f781bf', '#a65628', '#984ea3',
    #                                          '#999999', '#e41a1c', '#dede00']),
    #                                   int(max(y_pred) + 1))))
    # plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    # plt.show()

    modelfile = "blob2cluster_diag.bif"
    varNames = ["x","y"]
    bifparser = BifParser()
    net = bifparser.parse(modelfile)
    print("Before EM")
    print(net)
    
    learning_rate = 0.01
    batch_size = 100
    num_epochs = 1
    em = EMFramework(net)

#     i = 0
#     batchX = X[i*batch_size:min((i+1)*batch_size, len(X))]
#     em.stepwise_em_step(batchX, varNames, learning_rate, batch_size)
    start_time = time.time()
    for epoch in range(num_epochs):
        num_iters = int(math.floor(len(X))/batch_size)
        for i in range(num_iters):
            batchX = X[i*batch_size:min((i+1)*batch_size, len(X))]
            em.stepwise_em_step(batchX, varNames, learning_rate, batch_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time", elapsed_time)
    
    # get learned model parameters
    mus = np.zeros((n_centers, len(varNames)))
    covars = np.zeros((n_centers, len(varNames)))
    prob = np.zeros((n_centers,))
    for node in net.nodes:
        if isinstance(node, ContinuousBeliefNode):
            index = varNames.index(list(node.variable.variables)[0].name)
            for i in range(node.potential.size):
                    mus[i, index] = node.potential.get(i).mu[:]
                    covars[i, index] = node.potential.get(i).covar[:]
        elif isinstance(node, DiscreteBeliefNode):
            prob[:] = node.potential.parameter.prob
    
    print(prob)
    print(mus)
    print(covars)
    
    weights = np.array([0.4,0.6])
    means = np.array([[7.0,0.0],[7.0,9.0]])
    covariances = np.array([[1.0,1.0],[1.0,1.0]])
    gmm = GaussianMixture(n_components=n_centers)
    gmm.set_parameters(weights,means,covariances)
#     log_prob_norm, log_resp = gmm.stepwise_e_step(batchX)
#     gmm.stepwise_m_step(batchX, log_resp, learning_rate, batch_size)
    start_time = time.time()
    for epoch in range(num_epochs):
        num_iters = int(math.floor(len(X))/batch_size)
        for i in range(num_iters):
            batchX = X[i*batch_size:min((i+1)*batch_size, len(X))]
            log_prob_norm, log_resp = gmm.stepwise_e_step(batchX)
            gmm.stepwise_m_step(batchX, log_resp, learning_rate, batch_size)
    elapsed_time = time.time() - start_time
    print("Elapsed time", elapsed_time)
    print(gmm.weights_)
    print(gmm.means_)
    print(gmm.covariances_)
    

