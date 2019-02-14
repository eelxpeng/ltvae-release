import sys
sys.path.append("../../pyLTM")

import numpy as np
from jnius import autoclass
from jnius import cast

from pyltm.io import BifParser, BifWriter
from pyltm.learner import EMFramework
from pyltm.data import ContinuousDatacase
from pyltm.reasoner import NaturalCliqueTreePropagation, Evidence

# import necessary classes
PltmEast = autoclass("PltmEast")
GltmEast = autoclass("GltmEast")

def learn_latentTree(outpath, settingpath, datapath, initpath=None):
  if initpath is None:
    GltmEast.main(["-o", outpath, "-s", settingpath, "-c", "none", datapath])
  else:
    GltmEast.main(["-o", outpath, "-s", settingpath, "-c", "none", "-i", initpath, datapath])

def learn_pouchlatentTree(outpath, settingpath, datapath, initpath=None):
  if initpath is None:
    PltmEast.main(["-o", outpath, "-s", settingpath, "-c", "none", datapath])
  else:
    PltmEast.main(["-o", outpath, "-s", settingpath, "-c", "none", "-i", initpath, datapath])
    
class PouchLatentTree():
    def __init__(self, model_file, varNames):
        bifparser = BifParser()
        self._model = bifparser.parse(model_file)
        self.varNames = varNames
        self.em = EMFramework(self._model)

    def sample(self, z):
        """
        z: 2d numpy array
        """
        num = z.shape[0]
        out = []
        for i in range(num):
          x = self.pltm.sample(z[i].tolist())
          out.append(np.array(x).reshape((1, -1)))
        out = np.concatenate(out)
        return out

    def inference(self, x):
        """
        x: 2d numpy array
        """
        num = x.shape[0]
        latentVars = self._model.getInternalVariables()
        datacase = ContinuousDatacase.create(self.varNames)
        datacase.synchronize(self._model)
        ctp = NaturalCliqueTreePropagation(self._model)
        
        out = [None]*len(latentVars)
        datacase.putValues(x)
        evidence = datacase.getEvidence()
        ctp.use(evidence)
        ctp.propagate()
        for j in range(len(latentVars)):
          latent = np.exp(ctp.getMarginal(latentVars[j]).logprob)
          out[j] = latent

        return out

    def loglikelihood(self, x):
        """
        x: 2d numpy array, each row is one data case
        """
        assert(len(x.shape)==2)
        datacase = ContinuousDatacase.create(self.varNames)
        datacase.synchronize(self._model)
        ctp = NaturalCliqueTreePropagation(self._model)

        datacase.putValues(x)
        evidence = datacase.getEvidence()
        ctp.use(evidence)
        ctp.propagate()
        out = ctp.loglikelihood

        return out

    def grad(self, x):
        datacase = ContinuousDatacase.create(self.varNames)
        datacase.synchronize(self._model)
        ctp = NaturalCliqueTreePropagation(self._model)
        num, dim = x.shape
        out = np.zeros((num, dim), dtype=np.float32)

        datacase.putValues(x)
        evidence = datacase.getEvidence()
        ctp.use(evidence)
        ctp.propagate()

        tree = ctp.cliqueTree()
        gradient = np.zeros((num, dim))
        for node in self._model.getLeafNodes():
          clique = tree.getClique(node.variable)
          cliquePotential = clique.potential.clone()
          cliquePotential.normalize()

          moGpotential = node.potential.clone()
          
          vars = moGpotential.continuousVariables
          value = evidence.getValues(vars)

          b = np.zeros(len(vars))
          subgradient = np.zeros((num, len(vars)))
          for j in range(moGpotential.size):
            sol = np.linalg.solve(moGpotential.get(j).covar, (np.expand_dims(moGpotential.get(j).mu, axis=0) - value).T).T
            # subgradient += sol * np.expand_dims(cliquePotential.p[:, j], axis=1)
            subgradient += sol * np.expand_dims(np.exp(cliquePotential.logp[:, j]), axis=1)

          for j in range(len(vars)):
            index = datacase.variables.index(vars[j])
            gradient[:, index] = subgradient[:, j]

        return gradient

    def stepwise_em_step(self, X, learning_rate, batch_size, updatevar=True):
        self.em.stepwise_em_step(X, self.varNames, learning_rate, batch_size, updatevar=updatevar)

    def saveAsBif(self, filename):
        bifwriter = BifWriter()
        bifwriter.write(self._model, filename)

    def setCovariances(self, covar):
        '''covar: scalar'''
        for node in self._model.getLeafNodes():
          for i in range(node.potential.size):
            node.potential.get(i).covar[:] = np.eye(node.potential.dimension)*covar

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x[0], flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[0, ix]
    x[0, ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[0, ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[0, ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[0, ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[0, ix])
    it.iternext() # step to next dimension

  return grad

def rel_error(x, y):
    """returns relative error"""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__ == "__main__":
    varNames = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]
    batch_size = 10
    model = PouchLatentTree("../glass.bif", varNames)
    values = np.array([[1.51793,12.79,3.5,1.12,73.03,0.64,8.77,0,0]])
    loglikelihood = model.loglikelihood(values)
    print(loglikelihood)
    grad_analytic = model.grad(values)
    grad_numerical = eval_numerical_gradient(model.loglikelihood, 
      values, h=1e-5)
    print(grad_analytic)
    print(grad_numerical)
    print("Relative error: %f" % rel_error(grad_analytic, grad_numerical))
