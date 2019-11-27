#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:25:51 2018

@author: chemla
"""
import numpy as np
import torch
import torch.nn as nn
from . import VanillaDLGM 
import copy


class AudioMixtureDLGM(VanillaDLGM):
    def make_encoders(self, input_params, latent_params, *args, **kwargs):
        latent_params = copy.deepcopy(latent_params)
        latent_params[-1].append({'dim':len(latent_params[-1]), 'dist':torch.distributions.Normal, 'nn_lin':'sigmoid'})
        return super(AudioMixtureDLGM, self).make_encoders(input_params, latent_params, *args, **kwargs)

    def make_decoders(self, input_params, latent_params, hidden_params,  *args, **kwargs):
        return nn.ModuleList()

    def add_decoder(self, decoder):
        self.decoders.append(decoder.decoders)
        if not hasattr(self, 'decoder_callbacks'):
            self.decoder_callbacks = []
        self.decoder_callbacks.append(decoder.decode)
        
    def decode(self, z, y=None, sample=False, layer=-1, *args, **kwargs):
        outputs = []
        zs = [None]*len(z[-1])
        for l in range(len(zs)):
            zs[l] = z[-1][l]
        for i, z_i in enumerate(zs):
            outputs.append(self.decoder_callbacks[i](z_i, y=y, sample=sample, layer=layer, *args, **kwargs))
        return outputs

    def get_latent_projections(self, z_outs, weight_mode="angle"):  
        # get current device
        current_device = next(self.parameters()).device
        # projection callback
        def project(proj_matrix, v):
            return torch.bmm(proj_matrix.unsqueeze(0).repeat(v.shape[0],1,1), v.unsqueeze(-1)).squeeze(-1)
        # angle calculation callback
        def get_angle(z, proj):
            scalar_product = torch.sum(proj * z, dim=1)
            modulus = (torch.norm(proj, dim=1) * torch.norm(z, dim=1))
            product = scalar_product / modulus
            product = torch.where(torch.isnan(product), torch.tensor(0.0, requires_grad=True, device=current_device), product) 
            return torch.asin(torch.clamp(product, -1, 1))
        # join spaces
        dims = [z[0].shape[1] for z in z_outs]
        total_dim = sum(dims)
        dim_stride = np.cumsum([0]+dims);
        # decompose canonical base 
        eye_matrix = torch.eye(total_dim, requires_grad=False, device=current_device)
        bases = [eye_matrix[:, np.arange(dim_stride[i], dim_stride[i+1])] for i in range(len(dim_stride)-1)]
        projection_matrices = [ b.mm(torch.inverse(b.t().mm(b))).mm(b.t()) for b in bases]
        # get projected vectors
        z_full = torch.cat([z[0] for z in z_outs],1)
        projections = [project(projection_matrices[i], z_full) for i in range(len(projection_matrices))]
        # get angle
        weights = []
        for i, p in enumerate(projections):
            if weight_mode == "angle":
                angles = get_angle(z_full, p)
            elif weight_mode == "modulus":
                angles = torch.norm(p, dim=1)
            
            weights.append(angles/(np.pi/2))
        return weights

    def get_mixture(self, x_outs, mixture_params):
        # make mixture from individual outputs
        #weights = self.get_latent_projections(z_outs)
        weights = mixture_params['z_enc']
        mixture_out = torch.zeros_like(x_outs[0].mean, requires_grad=True, device=x_outs[0].mean.device)
        for i in range(len(x_outs)):
            mixture_out = mixture_out + weights[:, i].unsqueeze(1) * x_outs[i].mean
        return mixture_out, mixture_params['z_params_enc']

    def forward(self, x, options={}, *args, **kwargs):
        def denest_dict(nest_dict):
            keys = set()
            new_dict = {}
            for item in nest_dict:
                keys = keys.union(set(item.keys()))
            for k in keys:    
                new_dict[k] = [x[k] for x in nest_dict]
            return new_dict

        # encode
        x = self.format_input_data(x)
        enc_out = self.encode(x, *args, **kwargs)

        # separate mixture coefficients
        mixture_params = {'z_params_enc':enc_out[-1]['out_params'][-1], 'z_enc':enc_out[-1]['out'][-1]}
        enc_out[-1]['out_params'] = enc_out[-1]['out_params'][:-1]; enc_out[-1]['out'] = enc_out[-1]['out'][:-1]
        dec_in = denest_dict(enc_out)['out']

        # decode
        dec_out = self.decode(dec_in, *args, **kwargs)
        
        # format output
        enc_out = denest_dict(enc_out)
        out = {'z_params_enc': enc_out['out_params'], 'z_enc': enc_out['out'],
               'x_params': []}

        out_params = []
        for d in dec_out:
            x_params = d[0]
            if issubclass(type(x_params), list):
                x_params = [x['out_params'] for x in x_params]
            else:
                x_params = x_params['out_params']
            out_params.append(x_params)

        mixture_out, weights = self.get_mixture(out_params, mixture_params)
        out['x_params'] = mixture_out
        out['x_solo_params'] = out_params
        out['mixture_coeff'] = weights
#        del dec_in; del dec_out; del x; del enc_out; 
        return out

    def init_optimizer(self, optim_params={}):
        self.optim_params = optim_params
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-5})
        optimization_type = optim_params.get('mode', 'enc');
        assert optimization_type in ['full', 'enc']
        if optimization_type == 'enc':
            self.optimizers = {'default':getattr(torch.optim, alg)(self.encoders.parameters(), **optim_args)}   
        elif optimization_type == 'full':
            self.optimizers = {'default':getattr(torch.optim, alg)(self.parameters(), **optim_args)}   
        scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
        scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
        self.schedulers = {'default':getattr(torch.optim.lr_scheduler, scheduler)(self.optimizers['default'], **scheduler_args)} 

