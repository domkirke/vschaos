#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 11:47:20 2018

@author: chemla
"""
import pdb, torch
import torch.nn as nn
from . import VanillaDLGM
from ..modules.modules_hidden import HiddenModule
from ..distributions import Normal
from ..distributions.distribution_priors import IsotropicGaussian


class LadderVAE(VanillaDLGM):
    
    def init_modules(self, input_params, latent_params, hidden_params, *args, **kwargs):
        super(LadderVAE, self).init_modules(input_params, latent_params, hidden_params, *args, **kwargs)
        self.separate_decoder = kwargs.get('separate_decoder', False)
        if self.separate_decoder:
            self.td_encoders = self.make_decoders(latent_params[0], latent_params[1:])

    def make_decoders(self, input_params, latent_params, hidden_params, top_linear=True, *args, **kwargs):
        # no linear on top here
        decoders = nn.ModuleList()
        for i in reversed(range(0, len(latent_params))):
            if i == 0:
                phidden_dec = dict(hidden_params[0]); phidden_dec['batch_norm']=False
                decoders.append(HiddenModule(latent_params[0], phidden_dec, input_params,  name="dlgm_decoder_%d"%i))
            else:
                decoders.append(HiddenModule(latent_params[i], hidden_params[i], latent_params[i-1],  name="dlgm_decoder_%d"%i))
        return decoders

    def get_precision_weighted(self, dist_1, dist_2):
        assert type(dist_1)==Normal and type(dist_2)==Normal
        var_pw = 1 / (1/dist_1.variance + 1/dist_2.variance)
        mu_pw = (dist_1.mean/dist_1.variance + dist_2.mean/dist_2.variance)*var_pw
        return Normal(mu_pw, torch.sqrt(var_pw))

    def encode(self, x, y=None, sample=True, sample_mode="enc", *args, **kwargs):
        assert sample_mode in ["enc", "dec"]
        # get down-top encoding distributions
        enc_out = super(LadderVAE, self).encode(x, y=y, sample=sample, *args, **kwargs)

        # get latent activations from encoding path (all of them if taken from the encoders)
        if sample_mode == "enc":
            z_enc = [enc_out[i]['out'] for i in range(len(self.platent))]
        else:
            z_enc = [None]*len(self)
            z_enc[-1] = [enc_out[-1]['out']]

        # perform top-down passes
        if self.separate_decoder:
            decoders = self.td_decoders
        else:
            decoders = self.decoders

        dec_out = [None]*len(self)
        dec_out[-1] = enc_out[-1]
        enc_td_out = [None]*len(self)
        enc_td_out[-1] = enc_out[-1]
        for layer in range(len(self.platent)-2, -1, -1):
            if z_enc[layer] is None:
                z_enc[layer] = dec_out['out'][-1]
            decoder_idx = len(self.platent)-layer-2
            dec_out[layer] = decoders[decoder_idx](z_enc[layer+1])
            current_dist = {'out_params': self.get_precision_weighted(enc_out[layer]['out_params'], dec_out[layer]['out_params']), 'hidden':enc_out[layer]['hidden']}
            current_dist['out'] = current_dist['out_params'].rsample() if current_dist['out_params'].has_rsample else current_dist['out_params'].sample()
            enc_td_out[layer] = current_dist

        return enc_td_out



    def decode(self, enc_out, y=None, options={"sample": True}, *args, **kwargs):
        dec_out = [None]*(len(self)+1)
        dec_out[-1] = {'out': enc_out[-1]['out'], 'out_params':IsotropicGaussian(enc_out[-1]['out'].shape[0], self.platent[-1]['dim'])}
        for layer in range(len(self.platent)-1, -1, -1):
            dec_out[layer] = self.decoders[-layer-1](dec_out[layer+1]['out'])
        return dec_out

        
    def forward(self, x, options={}, *args, **kwargs):
        def denest_dict(nest_dict):
            keys = set()
            new_dict = {}
            for item in nest_dict:
                keys = keys.union(set(item.keys()))
            for k in keys:    
                new_dict[k] = [x[k] for x in nest_dict]
            return new_dict
        
        x = self.format_input_data(x)
        enc_out = self.encode(x, *args, **kwargs)
        dec_out = self.decode(enc_out, *args, **kwargs)[:-1]
        x_params = dec_out[0]['out_params']
        dec_out = denest_dict(dec_out[1:]) if len(dec_out) > 1 else {}
        enc_out = denest_dict(enc_out) if len(enc_out) > 1 else {}
        return {'x_params':x_params,
                'z_params_dec':dec_out.get('out_params'), 'z_dec':dec_out.get('out'),
                'z_params_enc':enc_out.get('out_params'), 'z_enc':enc_out.get('out')}

    
#    def forward(self, x, y=None, options={}, *args, **kwargs):
#        x = self.format_input_data(x)
#        z_params_enc, z_enc_dt = self.encode(x, y=y, *args, **kwargs)
#        x_params, (z_params_dec, z_params_enc_dt), (z_dec, z_enc_td) = self.decode(z_params_enc, y=y, *args, **kwargs)
#        return {'x_params':x_params, 
#                'z_params_dec':z_params_dec, 'z_dec':z_dec,
#                'z_params_enc':z_params_enc_dt, 'z_enc':z_enc_td}
        
