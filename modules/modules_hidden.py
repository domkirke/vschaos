#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 18:12:05 2018

@author: chemla
"""
import pdb
import torch
from torch import cat
import torch.nn as nn
import numpy as np

from . import init_module
from .modules_bottleneck import MLP, DLGMLayer
from .modules_distribution import get_module_from_density
from ..distributions.distribution_priors import IsotropicGaussian
from .flow import NormalizingFlow
from ..utils import checklist
from ..utils.misc import recgetattr, merge_dicts
from ..utils.utils_modules import Identity



# HiddenModule is an abstraction for inter-layer modules

# it has the following components:
#    hidden_modules : creates the hidden modules specified by phidden (can be a list)
#    out_modules : creates the modules corresponding to the output distributions specified by platent


class HiddenModule(nn.Module):
    default_module = MLP
    take_sequences = False
    dump_patches = True

    def __init__(self, pins, phidden=None, pouts=None, pflows=None, linked=True, *args, **kwargs):
        super(HiddenModule, self).__init__()
        if not issubclass(type(pins), list):
            pins = [pins]
        self.pins = pins; self.phidden = phidden; self.pouts = pouts
        
        # get hidden layers
        if phidden:
            self._hidden_modules = self.make_hidden_layers(pins, phidden, pouts=pouts, *args, **kwargs)
        else:
            self._hidden_modules = Identity()

        if not linked:
            assert len(phidden) == len(pouts)
        
        # get output layers
        self.out_modules = None
        self.linked = linked
        if pouts:
            pouts_input = phidden if phidden else pins
            self.out_modules = self.make_output_layers(pouts_input, pouts, *args, **kwargs)

    def make_hidden_layers(self, pins, phidden={"dim":800, "nlayers":2, 'label':None, 'conditioning':'concat'}, *args, **kwargs):
        if issubclass(type(phidden), list):
            return nn.ModuleList([self.make_hidden_layers(pins, dict(ph), *args, **kwargs) for ph in phidden])
        module_class = phidden.get('class', self.default_module)
        hidden_modules = module_class(pins, phidden, *args, **kwargs)
        return hidden_modules

    @property
    def hidden_modules(self):
        return self._hidden_modules

    @property
    def hidden_out_params(self, hidden_modules = None):
        hidden_modules = hidden_modules or self._hidden_modules
        if issubclass(type(hidden_modules), nn.ModuleList):
            params = []
            for i, m in enumerate(hidden_modules):
                if hasattr(hidden_modules[i], 'phidden'):
                    params.append(hidden_modules[i].phidden)
                else:
                    params.append(checklist(self.phidden, n=len(hidden_modules))[i])
            return params
        else:
            if hasattr(hidden_modules, 'phidden'):
                return checklist(hidden_modules.phidden)[-1]
            else:
                return checklist(checklist(self.phidden)[0])[-1]

    def make_output_layers(self, pins, pouts, *args, **kwargs):
        '''returns output layers with resepct to the output distribution
        :param in_dim: dimension of input
        :type in_dim: int
        :param pouts: properties of outputs
        :type pouts: dict or [dict]
        :returns: ModuleList'''
        out_modules = []
        pouts = checklist(pouts)

        current_hidden_params = checklist(self.hidden_out_params, n=len(pouts))
        for i, pout in enumerate(pouts):
            if issubclass(type(pout),  dict):
                if issubclass(type(pins), list):
                    if self.linked:
                        # each output distribution has a separated MLP
                        input_dim = checklist(sum([x['dim'] for x in pins]), copy=True)[-1]
                        current_encoders = self.hidden_modules
                    else:
                        # output distributions share the same MLP
                        input_dim = checklist(pins[i]['dim'], copy=True)[-1]
                        current_encoders = self.hidden_modules[i]
                else:
                    input_dim = checklist(pins['dim'])[-1]
                    current_encoders = self.hidden_modules

                out_modules.append(self.get_module_from_density(pout["dist"])(current_hidden_params[i], pout, hidden_module=current_encoders, **kwargs))
            else:
                # if pout is just an int
                out_modules.append(nn.Linear(input_dim, pout))
        out_modules = nn.ModuleList(out_modules)
        return out_modules

    def get_module_from_density(self, dist):
        return get_module_from_density(dist)

    def forward_hidden(self, x, y=None, *args, **kwargs):
        if issubclass(type(self.hidden_modules), list):
            hidden_out = [h(x[i], y=y, sample=True, *args, **kwargs) for i,h in enumerate(self.hidden_modules)]
        else:
            hidden_out = self.hidden_modules(x, y=y, *args, **kwargs)
        
        if self.linked and issubclass(type(self.hidden_modules), list): 
            hidden_out = torch.cat(tuple(hidden_out), 1)
        else:
            hidden_out = hidden_out
        return hidden_out
    
    def forward_params(self, hidden, y=None, *args, **kwargs):
        # get distirbutions from distribution module
        z_dists = []
        for i, out_module in enumerate(self.out_modules):
            if issubclass(type(self.hidden_modules), nn.ModuleList):
                indices = None
                if out_module.requires_deconv_indices:
                    indices = self.hidden_modules[i].get_pooling_indices()
                if self.linked: 
                    z_dists.append(out_module(hidden, indices=indices))
                else:
                    z_dists.append(out_module(hidden[i], indices=indices))
            else: 
                requires_deconv_indices = recgetattr(out_module, 'requires_deconv_indices')
                indices = None
                if requires_deconv_indices:
                    indices = self.hidden_modules.get_pooling_indices()

                if issubclass(type(hidden), list):
                    z_dists.append(out_module(hidden[i], indices=checklist(indices)[i]))
                else:
                    z_dists.append(out_module(hidden, indices=indices))

        if not issubclass(type(self.pouts), list):
            z_dists = z_dists[0]
        return z_dists
        

    def forward(self, x, y=None, sample=True, return_hidden=False, *args, **kwargs):
        # get hidden representations
        out = {}
        hidden_out = self.forward_hidden(x, y=y, *args, **kwargs)
        if return_hidden:
            out['hidden'] = hidden_out
        if self.out_modules is not None:
            # get output distributions
            out['out_params'] = self.forward_params(hidden_out, y=y, *args, **kwargs)

        return out


class DLGMModule(HiddenModule):
    r"""
    Specific decoding module for Deep Latent Gaussian Models
    """
    def get_module_from_density(self, dist):
        return DLGMLayer
    
    def sample(self, h, eps, *args, **kwargs):
        z = []
        for i, out_module in enumerate(self.out_modules):
            if issubclass(type(self.hidden_modules), nn.ModuleList):
                if self.linked: 
                    z.append(out_module(h, eps))
                else:
                    z.append(out_module(h[i], eps[i]))
            else: 
                z.append(out_module(h, eps))
        # sum
        z = [current_z[0] + current_z[1] for current_z in z]
        if not issubclass(type(self.pouts), list):
            z = z[0]
        return z
    
    def forward_params(self, h, *args, **kwargs):
        batch_size = h[0].shape[0] if issubclass(type(h), list) else h.shape[0]
        if issubclass(type(self.pouts), list):
            target_shapes = [(batch_size, self.pouts[i]['dim']) for i in range(len(self.pouts))]
            params = [IsotropicGaussian(*target_shapes[i], device=h.device) for i in range(len(out['out_params']))]
        else:
            params = IsotropicGaussian(batch_size, self.pouts['dim'], device=h.device)
        return params

    def forward(self, x, y=None, sample=True, *args, **kwargs):
        # get hidden representations
        out = {'hidden': self.forward_hidden(x, y=y, *args, **kwargs)}
        if self.out_modules is not None:
            # get output distributions
            out['out_params'] = self.forward_params(out['hidden'], y=y, *args, **kwargs)
            # get samples
            if sample:
                out['out'] = self.sample(out['hidden'], y=y, *args, **kwargs)
        return out
