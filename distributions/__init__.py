# -*- coding: utf-8 -*-
"""
    The ``distributions`` module
    ========================

    This package provides additional features for the native Pytorch Distribution object, implement new
        distributions, new divergences and predefined priors for Bayesian inference

    :Example:

    >>> from vschaos import distributions as dist
    >>> n = dist.Normal(torch.zeros(10,10), torch.ones(10, 10))
    >>> n = n.reshape(100)

    Subpackages available
    ---------------------

        * priors

    Comments and issues
    ------------------------

        None for the moment

    Contributors
    ------------------------

    * Philippe Esling       (esling@ircam.fr)

"""
import torch
from torch.distributions import *
from .distribution_class import *
from .distribution_process import *
from . import distribution_priors as priors
from .distribution_flow import *



# adds manipulation function for distributions

def distribution_getitem(self, slice):
    if type(self) == Normal:
        return Normal(self.mean.__getitem__(slice), self.stddev.__getitem__(slice))
    elif type(self) == Bernoulli:
        return Bernoulli(self.mean.__getitem__(slice))
    elif type(self) == Categorical:
        return Categorical(logits=self.logits(slice))

def distribution_indexselect(self, dim, idx):
    idx = idx.to(self.mean.device)
    if type(self) == Normal:
        return Normal(torch.index_select(self.mean, dim, idx), torch.index_select(self.stddev, dim, idx))
    elif type(self) == Bernoulli:
        return Bernoulli(torch.index_select(self.mean, dim, idx))
    elif type(self) == Categorical:
        return Categorical(logits=torch.index_select(self.mean, dim, idx))

def distribution_to(self, *args, **kwargs):
    if issubclass(type(self), Bernoulli):
        return Bernoulli(self.mean.to(*args, **kwargs))
    if issubclass(type(self), Normal):
        return Normal(self.mean.to(*args, **kwargs), self.stddev.to(*args, **kwargs))
    else:
        raise NotImplementedError

def distribution_reshape(self, *shape):
    if issubclass(type(self), Bernoulli):
        return Bernoulli(self.mean.reshape(shape))
    if issubclass(type(self), Normal):
        return Normal(self.mean.reshape(shape), self.stddev.reshape(shape))
    else:
        raise NotImplementedError

def distribution_view(self, *shape, contiguous=True):
    if issubclass(type(self), Bernoulli):
        return Bernoulli(self.mean.contiguous().view(shape))
    if issubclass(type(self), (Normal, MultivariateNormal, RandomWalk)):
        return type(self)(self.mean.contiguous().view(shape), self.stddev.contiguous().view(shape))
    if issubclass(type(self), Categorical):
        return type(self)(probs=self.probs.contiguous().view(shape))
    else:
        raise NotImplementedError

def distribution_squeeze(self):
    if issubclass(type(self), Bernoulli):
        return Bernoulli(self.mean.squeeze())
    if issubclass(type(self), (Normal, MultivariateNormal, RandomWalk)):
        return type(self)(self.mean.squeeze(), self.stddev.squeeze())
    if issubclass(type(self), Categorical):
        return type(self)(probs=self.probs.squeeze())
    else:
        raise NotImplementedError

def scramble(tensor, dim=-1):
    slices = [slice(None)]*len(tensor.shape)
    slices[dim] = torch.randperm(tensor.shape[dim])
    return tensor.__getitem__(slices)

def distribution_scramble(self, dim=-1):
    if issubclass(type(self), Bernoulli):
        return Bernoulli(scramble(self.mean, dim))
    if issubclass(type(self), (Normal, MultivariateNormal, RandomWalk)):
        return type(self)(scramble(self.mean, dim), scramble(self.stddev, dim))
    if issubclass(type(self), Categorical):
        return type(self)(probs=scramble(self.probs, dim))
    else:
        raise NotImplementedError


Distribution.__getitem__ = distribution_getitem
Distribution.reshape = distribution_reshape
Distribution.view = distribution_view
Distribution.squeeze = distribution_squeeze
Distribution.scramble = distribution_scramble
Distribution.index_select = distribution_indexselect
Distribution.to = distribution_to
Distribution.class_dependant = False
Distribution.requires_preflow = False

# just a casuality to turn deterministic stuff as distributions

class Empirical(Distribution):
    has_rsample = True
    def __init__(self, tensor):
        self.tensor = tensor

    def log_prob(self, value):
        if value != self.tensor:
            return 0.
        else:
            return torch.tensor(1.)

    def sample(self, sample_shape=torch.Size()):
        return self.tensor

    def rsample(self, sample_shape=torch.Size()):
        return self.tensor

