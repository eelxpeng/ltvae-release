import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import logging
from lib.plt_pyltm import PouchLatentTree
# import pydevd

class LatentTreeModule(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.latObj = None

    def updateLatentTree(self, model_file, varNames):
        self.latObj = PouchLatentTree(model_file, varNames)

    def forward(self, input):
        ops = LatentTreeOps(self.latObj)
        return ops(input)

class LatentTreeOps(torch.autograd.Function):
    def __init__(self, latObj):
        super(self.__class__, self).__init__()
        self.latObj = latObj
        self.use_cuda = torch.cuda.is_available()

    def forward(self, input):
        # print("forward")
        numpy_input = input.data.cpu().numpy()
        # logging.info("latentTree input:")
        # logging.info(np.array2string(numpy_input, precision=4, separator=',', suppress_small=True))
        value = self.latObj.loglikelihood(numpy_input)
        self.save_for_backward(input)
        out = torch.FloatTensor(value)
        if self.use_cuda:
            out = out.cuda()
        return out

    def backward(self, grad_output):
        # print("backward")
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        numpy_go = grad_output.cpu().numpy().reshape(-1, 1)
        # print(numpy_go)
        # print(numpy_go.shape)
        input = self.saved_tensors[0]
        numpy_input = input.data.cpu().numpy()
        result = self.latObj.grad(numpy_input) * numpy_go
        # logging.info("latentTree grad:")
        # logging.info(np.array2string(result, precision=4, separator=',', suppress_small=True))
        # logging.info("max: %s, min: %s" % (result.max(), result.min()))
        # nanidx = np.argwhere(np.isnan(result))
        # if len(nanidx)>0:
        #     logging.info(np.array2string(nanidx))
        #     logging.info("Input:")
        #     logging.info(np.array2string(numpy_input[nanidx[:, 0]], precision=4, separator=',', suppress_small=True))
        out = torch.FloatTensor(result)
        if self.use_cuda:
            out = out.cuda()
        return out
        