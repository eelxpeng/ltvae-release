import numpy as np
import math

def logsumexp(value, axis=None, keepdims=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if axis is not None:
        m = np.max(value, axis=axis, keepdims=True)
        value0 = value - m
        if keepdims is False:
            m = m.squeeze(axis)
        return m + np.log(np.sum(np.exp(value0),
                                       axis=axis, keepdims=keepdims))
    else:
        m = np.max(value)
        sum_exp = np.sum(np.exp(value - m))
        if isinstance(sum_exp, np.ScalarType):
            return m + math.log(sum_exp)
        else:
            return m + np.log(sum_exp)
