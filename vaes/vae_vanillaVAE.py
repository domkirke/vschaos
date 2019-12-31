#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:38:11 2017

@author: chemla
"""

import pdb
import torch.nn as nn
import torch.optim

from ..modules.modules_hidden import  HiddenModule
from ..utils.misc import GPULogger, denest_dict, apply, apply_method, apply_distribution, print_stats
from . import AbstractVAE 

logger = GPULogger(verbose=False)

class VanillaVAE(AbstractVAE):
    HiddenModuleClass = [HiddenModule, HiddenModule]
    # initialisation of the VAE
    def __init__(self, input_params, latent_params, hidden_params = {"dim":800, "nlayers":2}, hidden_modules=None, *args, **kwargs):
        self.set_hidden_modules(hidden_modules)
        super(VanillaVAE, self).__init__(input_params, latent_params, hidden_params, *args, **kwargs)
        
    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        encoders = nn.ModuleList()
        for layer in range(len(latent_params)):
            if layer==0:
                encoders.append(self.make_encoder(input_params, latent_params[0], hidden_params[0], name="vae_encoder_%d"%layer, *args, **kwargs))
            else:
                encoders.append(self.make_encoder(latent_params[layer-1], latent_params[layer], hidden_params[layer], name="vae_encoder_%d"%layer, *args, **kwargs))
        return encoders

    @classmethod
    def make_encoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):               
        kwargs['name'] = kwargs.get('name', 'vae_encoder')
#        ModuleClass = hidden_params.get('class', DEFAULT_MODULE)
#        module = latent_params.get('shared_encoder') or ModuleClass(input_params, latent_params, hidden_params, *args, **kwargs)
        module_class = kwargs.get('module_class', cls.HiddenModuleClass[0])
        module = module_class(input_params, hidden_params, latent_params, *args, **kwargs)
        return module
    
    def make_decoders(self, input_params, latent_params, hidden_params, extra_inputs=[], *args, **kwargs):
        decoders = nn.ModuleList()
        for layer in range(len(latent_params)):
            if layer==0:
                new_decoder = self.make_decoder(input_params, latent_params[0], hidden_params[0], name="vae_decoder_%d"%layer, encoder = self.encoders[layer], *args, **kwargs)
            else:
                new_decoder = self.make_decoder(latent_params[layer-1], latent_params[layer], hidden_params[layer], name="vae_decoder_%d"%layer, encoder=self.encoders[layer], *args, **kwargs)
            decoders.append(new_decoder)
        return decoders
    
    @classmethod
    def make_decoder(cls, input_params, latent_params, hidden_params, *args, **kwargs):
        kwargs['name'] = kwargs.get('name', 'vae_decoder')
#        ModuleClass = hidden_params.get('class', DEFAULT_MODULE)
#        module = hidden_params.get('shared_decoder') or ModuleClass(latent_params, input_params, hidden_params, *args, **kwargs)
        module_class = kwargs.get('module_class', cls.HiddenModuleClass[1])
        module = module_class(latent_params, hidden_params, input_params, make_flows=False, *args, **kwargs)
        return module

    def set_hidden_modules(self, hidden_modules):
        if hidden_modules is None:
            return
        if issubclass(type(hidden_modules), type):
            self.HiddenModuleClass = [hidden_modules, hidden_modules]
        elif issubclass(type(hidden_modules), list):
            self.HiddenModuleClass = [hidden_modules[0], hidden_modules[1]]


    # processing methods

    def encode(self, x, y=None, sample=True, from_layer=0, *args, **kwargs):
        ins = x; outs = []
        for layer in range(from_layer, len(self.platent)):
            module_out = self.encoders[layer](ins, y=y, *args, **kwargs)
            out_params = module_out['out_params']
            if module_out.get('out') is None:
                try:
                    ins = apply_method(out_params, 'rsample')
                except NotImplementedError:
                    ins = apply_method(out_params, 'sample')
                if issubclass(type(ins), tuple):
                    ins, z_preflow = ins
                    module_out['out_preflow'] = z_preflow
                module_out['out'] = ins
            outs.append(module_out)
            if issubclass(type(module_out), list):
                ins = torch.cat(ins, dim=-1)
        return outs
        
    def decode(self, z, y=None, sample=True, from_layer=-1, *args, **kwargs):
        if from_layer < 0:
            from_layer += len(self.platent)
        ins = z; outs = []
        for i,l in enumerate(reversed(range(from_layer+1))):
            module_out = self.decoders[l](ins, y=y, *args, **kwargs)
            out_params = module_out['out_params']
            if module_out.get('out') is None:
                try:
                    ins = apply_method(out_params, 'rsample')
                except NotImplementedError:
                    ins = apply_method(out_params, 'sample')
                if issubclass(type(ins), tuple):
                    ins, z_preflow = ins
                    module_out['out_preflow'] = z_preflow
                module_out['out'] = ins
            outs.append(module_out)
            if issubclass(type(module_out), list):
                ins = torch.cat(ins, dim=-1)
        outs = list(reversed(outs))
        return outs

    def forward(self, x, y=None, options={}, *args, **kwargs):
        x = self.format_input_data(x, requires_grad=False)
        # logger("data formatted")
        enc_out = self.encode(x, y=y, *args, **kwargs)
        #print_stats(enc_out[-1]['out_params'].mean, "latent mean")
        #print_stats(enc_out[-1]['out_params'].stddev, "latent std")

        logger("data encoded")
        dec_out = self.decode(enc_out[-1]['out'], y=y, *args, **kwargs)
        logger("data decoded")
        
        x_params = dec_out[0]['out_params']
        dec_out = denest_dict(dec_out[1:]) if len(dec_out) > 1 else {}
        enc_out = denest_dict(enc_out)       
        logger("output formatted")

        return {'x_params':x_params,
                'z_params_dec':dec_out.get('out_params'), 'z_dec':dec_out.get('out'),
                'z_params_enc':enc_out['out_params'], 'z_enc':enc_out['out'],
                "z_preflow_enc":enc_out.get('out_preflow'),"z_preflow_dec":dec_out.get('out_preflow')}

    
    # define optimizer
    def init_optimizer(self, optim_params, init_scheduler=True):
        self.optim_params = optim_params
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-3})
        optimization_mode = optim_params.get('mode', 'full')

        optimizer = getattr(torch.optim, alg)([{'params':self.encoders.parameters()}], **optim_args)
        if optimization_mode == 'full':
            optimizer.add_param_group({'params':self.decoders.parameters()})
        self.optimizers = {'default':optimizer}
        if init_scheduler:
            self.init_scheduler(optim_params)

    def init_scheduler(self, optim_params):
        scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
        scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
        self.schedulers = {'default':getattr(torch.optim.lr_scheduler, scheduler)(self.optimizers['default'], **scheduler_args)} 
        
        
    def optimize(self, loss, options={}, retain_graph=False, *args, **kwargs):
        # optimize
        self.optimizers['default'].step()

    def schedule(self, loss, options={}):
        self.schedulers['default'].step(loss)

        
    # define losses 
    

