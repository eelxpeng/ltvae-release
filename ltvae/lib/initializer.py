import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data
from torch.autograd import Variable
import math

def weights_variance_scaling_init(m):
    if isinstance(m, nn.Linear):
        variance_scaling_initializer(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def weights_xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def variance_scaling_initializer(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    """Fills the input Tensor or Variable with values according to the method
    described in "Understanding the difficulty of training deep feedforward
    neural networks" - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`U(-a, a)` where
    :math:`a = gain \\times \sqrt{2 / (fan\_in + fan\_out)} \\times \sqrt{3}`.
    Also known as Glorot initialisation.

    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        gain: an optional scaling factor

    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('relu'))
    """
    # if isinstance(tensor, Variable):
    #     variance_scaling_initializer(tensor.data, scale=scale, mode=mode, distribution=distribution)
    #     return tensor

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
      # Count only number of input connections.
      n = fan_in
    elif mode == 'fan_out':
      # Count only number of output connections.
      n = fan_out
    elif mode == 'fan_avg':
      # Average number of inputs and output connections.
      n = (fan_in + fan_out) / 2.0
    if distribution=="normal":
        std = math.sqrt(scale / n)
        return tensor.normal_(0, std)
    if distribution == "uniform":
        limit = math.sqrt(3.0 * scale / n)
        return tensor.uniform_(-limit, limit)
