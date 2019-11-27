#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:36:44 2017

@author: chemla
"""
import torch
import torch.nn as nn
import sys, pdb

from ..modules.modules_hidden import HiddenModule, DLGMModule
from . import VanillaVAE 
from ..distributions import priors
from ..utils.misc import GPULogger

logger = GPULogger(verbose=False)

class VanillaDLGM(VanillaVAE):
    
    def __init__(self, input_params, latent_params, hidden_params={"dim": 800, "layers": 2}, *args, **kwargs):
        super(VanillaDLGM, self).__init__(input_params, latent_params, hidden_params, *args, **kwargs)
                
    def make_encoders(self, input_params, latent_params, hidden_params, *args, **kwargs):
        encoders = nn.ModuleList()
        for i in range(len(latent_params)):
            if i == 0:
                encoders.append(self.make_encoder(input_params, latent_params[i], hidden_params[i], name="dlgm_encoder_%d"%i, *args, **kwargs))
            else:
                encoders.append(self.make_encoder(hidden_params[i-1], latent_params[i], hidden_params[i], name="dlgm_encoder_%d"%i, *args, **kwargs))
        return encoders

    def make_decoders(self, input_params, latent_params, hidden_params, top_linear=True, *args, **kwargs):   
        decoders = nn.ModuleList()
        # add top linear transformation
        for i in range(0, len(latent_params)):
            if i == 0:
                phidden_dec = dict(hidden_params[0]); phidden_dec['batch_norm']=False
                decoders.append(HiddenModule(latent_params[0], phidden_dec, input_params, name="dlgm_decoder_%d"%i))
            else:
                decoders.append(DLGMModule(latent_params[i], hidden_params[i], latent_params[i-1], name="dlgm_decoder_%d"%i))
        top_dim = latent_params[-1]['dim'] if not issubclass(type(latent_params[-1]), list)  else sum([x['dim'] for x in latent_params[-1]])
        decoders.append(nn.Linear(top_dim, top_dim))
        return decoders

    # Process routines
    def encode(self, x, y=None, sample=True, *args, **kwargs):
        ins = x; outs = []
        for layer in range(len(self.platent)):
            module_out = self.encoders[layer](ins, *args, **kwargs)
            outs.append(module_out)
            ins = module_out['hidden']
        return outs
        
    def decode(self, z, y=None, sample=True, layer=-1, *args, **kwargs):
        assert layer != 0
        if layer < 0:
            layer += len(self.platent) + 1
            
        if not issubclass(type(z), list):
            z = [z]
        n_batches = z[0].size(0)
        
        # if starting from top layer, we pass the latent postion through the top linear module (eps_L)
        #       otherwise, the first given latent variable is the hidden variable h_l
        z = list(reversed(z))
        
        current_z_idx = 0
        outs = []
        for i, l in enumerate(reversed(range(1, layer+1))):
            # during the fist step of the loop, z[0] can be whether eps_L or h_l
            if i == 0:
                if layer == len(self.platent):
                    # first z is eps_L
                    previous_h = self.decoders[-1](z[0])
                    current_z_idx += 1
                else:
                    # first z is h_l
                    previous_h = z[0]
                    current_z_idx += 1
            else:
                if sample:
                    if current_z_idx < len(z):
                        eps = z[current_z_idx]
                    else:
                        eps = priors.IsotropicGaussian(self.platent[l-1]['dim'], device=previous_h.device)(n_batches)
                        eps = eps.to(previous_h.device)
                else:
                    eps = torch.zeros(n_batches, self.platent[l-1]['dim'], requires_grad=True, device=previous_h.device)
                current_out = self.decoders[l](previous_h, eps=eps)
                outs.append(current_out)
                previous_h = current_out['out']
                current_z_idx += 1
        final_output = self.decoders[0](previous_h)
        outs.append(final_output)
        outs = list(reversed(outs))
        return outs

    def forward(self, x, options={}, *args, **kwargs):
        def denest_dict(nest_dict):
            keys = set()
            new_dict = {}
            for item in nest_dict:
                keys = keys.union(set(item.keys()))
            for k in keys:    
                new_dict[k] = [x[k] for x in nest_dict]
            return new_dict
        
        logger('init')
        x = self.format_input_data(x)
        enc_out = self.encode(x, *args, **kwargs)
        logger("data encoded")
        dec_in = denest_dict(enc_out)['out']
        
        dec_out = self.decode(dec_in, *args, **kwargs)
        logger("data deocded")
        x_params = dec_out[0]
        if issubclass(type(x_params), list):
            x_params = [x['out_params'] for x in x_params]
        else:
            x_params = x_params['out_params']
            
        dec_out = denest_dict(dec_out[1:]) if len(dec_out) > 1 else {}
        enc_out = denest_dict(enc_out)       
        logger("data formatted")
        return {'x_params':x_params, 
                'z_params_dec': dec_out.get('out_params'), 'z_dec': dec_out.get('out'),
                'z_params_enc': enc_out['out_params'], 'z_enc': enc_out['out']}
