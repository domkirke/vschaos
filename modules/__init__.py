import torch.nn as nn



MLP_DEFAULT_NNLIN = "ELU"
CONV_DEFAULT_NNLIN = "ELU"
DEFAULT_INIT = nn.init.xavier_normal_
CONDITIONING_HASH = ['concat']


def get_init(nn_lin):
    if nn_lin=="ReLU":
        return 'relu'
    elif nn_lin=="TanH":
        return 'tanh'
    elif nn_lin=="LeakyReLU":
        return 'leaky_relu'
    elif nn_lin=="conv1d":
        return "conv1d"
    elif nn_lin=="cov2d":
        return "conv2d"
    elif nn_lin=="conv3d":
        return "conv3d"
    elif nn_lin=="Sigmoid":
        return "sigmoid"
    else:
        return "linear"
    
def init_module(module, nn_lin=MLP_DEFAULT_NNLIN, method=DEFAULT_INIT):
    gain = nn.init.calculate_gain(get_init(nn_lin))
    if type(module)==nn.Sequential:
        for m in module:
            init_module(m, nn_lin=nn_lin, method=method)
    if type(module)==nn.Linear:
        method(module.weight.data, gain)
        nn.init.zeros_(module.bias)

class Identity(nn.Module):
    def __call__(self, *args, **kwargs):
        return args

class Sequential(nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self._modules.values():
            input = module(input, *args, **kwargs)
        return input


def flatten(x, dim=1):
    if len(x.shape[dim:]) != 1:
        if not x.is_contiguous():
            x = x.contiguous()
        return x.view(*tuple(x.shape[:dim]), np.cumprod(x.shape[dim:])[-1])
    else:
        return x

from . import flow
from .modules_bottleneck import *
from .modules_convolution import *
from .modules_distribution import BernoulliLayer, GaussianLayer, get_module_from_density, CategoricalLayer
from .modules_hidden import *
from .modules_recurrent import *
from .modules_prediction import *

