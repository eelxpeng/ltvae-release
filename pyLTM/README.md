# pyLTM library

This repository tries to provide Python library for Bayesian latent tree model. This imlementation is based on commonly used Python library such as numpy, scipy, etc. The original java library with structure learning for latent tree models is [HLTA](https://github.com/kmpoon/hlta) and [Pouch latent tree model](https://github.com/kmpoon/pltm-east).

The functionality implemented include
* Gaussian latent tree model
* Clique Tree Propagation
* Parser for bif model file

A working example is shown as follows, where model file is under example/

```
from pyltm.model import Gltm
from pyltm.reasoner import NaturalCliqueTreePropagation
from pyltm.io import BifParser
from pyltm.data import ContinuousDatacase

modelfile = "continuoustoy.bif"
varNames = ["x"]
data = [0]

bifparser = BifParser()
net = bifparser.parse(modelfile)

# set up evidence
datacase = ContinuousDatacase.create(varNames)
datacase.synchronize(net)
datacase.putValues(data)
evidence = datacase.getEvidence()

ctp = NaturalCliqueTreePropagation(net)
print(ctp._tree)
ctp.use(evidence)
ctp.propagate()
loglikelihood = ctp.loglikelihood
print("Loglikelihood: ", loglikelihood)

```

# contributors
* Xiaopeng Li
* Jiqing Wen