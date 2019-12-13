# -*-coding:utf-8 -*-

"""
    The ``datasets`` module
    ========================

    This package contains all datasets classes

    :Example:

    >>> from data.sets import DatasetAudio
    >>> DatasetAudio()

    Subpackages available
    ---------------------

        * Generic
        * Audio
        * Midi
        * References
        * Time Series
        * Pytorch
        * Tensorflow

    Comments and issues
    ------------------------

        None for the moment

    Contributors
    ------------------------

    * Philippe Esling       (esling@ircam.fr)

"""

# info
__version__ = "1.0"
__author__  = "chemla@ircam.fr", "esling@ircam.fr"
__date__    = ""
__all__     = ["criterions", "data", "distributions", "modules", "monitor", "train", "utils", "vaes", "DataParallel"]

import torch, pdb
from torch.nn.parallel.scatter_gather import scatter_kwargs
torch.manual_seed(0)

#TODO this is a hack for GPU training where tkinter is not installed; do not deploy!!
#if torch.cuda.is_available():
#    import matplotlib
#    matplotlib.use('agg')

# overriding DataParallel to allow distribution parallelization
# tests for sub-commit

DataParallel = torch.nn.DataParallel

def gather(self, *args, **kwargs):
    return utils.gather(*args, **kwargs)

def __getattr(self, attribute):
    try:
        return super(DataParallel, self).__getattr__(attribute)
    except AttributeError:
        return getattr(self.module, attribute)

def scatter(self, inputs, kwargs, device_ids):
    return utils.scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


def load(path, **kwargs):
    loaded_data = torch.load(path, **kwargs)
    return loaded_data


DataParallel.gather = gather
DataParallel.__getattr__ = __getattr
DataParallel.scatter = scatter
from . import utils
from . import distributions
from . import criterions
from . import data
#from . import misc
from . import modules
from . import monitor
from . import vaes
from . import train
from itertools import chain




