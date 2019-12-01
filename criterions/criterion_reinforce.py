#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:10:00 2018

@author: chemla
"""
import torch, torch.nn as nn, math, pdb
from numpy import cumprod
from ..modules import modules_bottleneck as bt
from ..modules import modules_convolution as conv
from ..modules.modules_recurrent import RNNLayer
from .criterion_criterion import Criterion
from .criterion_adversarial import Adversarial
from collections import OrderedDict
from ..utils.misc import checklist, checktuple, print_module_stats, flatten_seq_method


######################################################################
# -----    Modules for reinforcement (initialized to identity)
######################################################################


class MLPReinforcementModule(torch.nn.Module):
    def __init__(self, in_params, module_params, latent_params=None, recurrent_z=False):
        super(MLPReinforcementModule, self).__init__()
        n_layers = module_params.get('nlayers', 1)
        layer = module_params.get('layer_class', bt.MLPGatedLayer)
        hidden_dims = checklist(module_params.get('dim', []), n_layers)
        modules = []
        # retrieve input params
        input_dim = in_params['dim']
        if latent_params is not None:
            lp = checklist(latent_params)[0]
            input_dim += sum([k['dim'] for k in checklist(lp)])
        output_dim = in_params['dim']

        for i in range(n_layers):
            dim_in = input_dim if i==0 else hidden_dims[i]
            dim_out = output_dim if i==n_layers-1 else hidden_dims[i]
            modules.append(layer(dim_in, dim_out, bias=False, batch_norm=False, nn_lin=module_params.get('nn_lin')))
            #torch.nn.init.normal_(modules[i].hidden.weight)
        self.layers = nn.ModuleList(modules)
        #self.nn_lin = module_params.get('nn_lin')


    @flatten_seq_method
    def forward(self, input, z=None):
        input_shape = input.shape
        if len(input_shape) > 2:
            input = input.view(input_shape[0], cumprod(input_shape[1:])[-1])
        if z is not None:
            z_input = z[0]
            if len(z_input.shape) > 2:
                z_input = z_input.contiguous().view(z_input.shape[0]*z_input.shape[1], *z_input.shape[2:])
            pdb.set_trace()
            input = torch.cat([input, z_input], dim=-1)
        for i, module in enumerate(self.layers):
            input = module(input)
            if len(self.layers) > 1 and i < len(self.layers) - 1:
                input = torch.nn.functional.relu(input)
        if input_shape != input.shape:
            input = input.view(*input_shape)
        '''
        if self.nn_lin:
            input = getattr(torch.nn.functional, self.nn_lin)(input)
        '''
        return input


class ResidualMLPReinforcementModule(MLPReinforcementModule):
    init_value = 1e-5
    def __init__(self, in_params, module_params):
        module_params['nn_lin'] = None
        super(ResidualMLPReinforcementModule, self).__init__(in_params, module_params)
        '''
        for i in range(len(self._modules)):
            self.layers[i].weight.data = self.layers[i].weight.data * self.init_value
        '''

    def forward(self, input, z=None):
        out = super().forward(input, z=z)
        return out + input


class ConvReinforcementModule(torch.nn.Module):
    conv_hash = {1:nn.Conv1d, 2:nn.Conv2d, 3:nn.Conv3d}
    conv_fun_hash = {1:torch.nn.functional.conv1d, 2:torch.nn.functional.conv2d, 3:torch.nn.functional.conv3d}

    def __init__(self, in_params, module_params, latent_params=None):
        super(ConvReinforcementModule, self).__init__()
        self.channels = module_params.get('channels', [16,16])
        self.input_channels = in_params.get('channels') or 1
        self.conv_dim = len(checklist(in_params['dim']))
        self.kernel_size = checklist(module_params.get('kernel_size', [3,3,3]), len(self.channels))
        self.paddings = [0 if k == 0 else math.ceil(k/2) for k in self.kernel_size]
        modules = []
        self.platent = latent_params
        nlayers = len(self.channels)+1
        for i in range(nlayers):
            if i==0:
                input_dim = self.input_channels; output_dim = self.channels[0]
            elif i == len(self.channels):
                input_dim = self.channels[-1]; output_dim = self.input_channels
            else:
                input_dim = self.channels[i-1]; output_dim = self.channels[i]

            if latent_params is None: 
                modules.append(self.conv_hash[self.conv_dim](input_dim, output_dim, kernel_size=self.kernel_size[i], padding=self.paddings[i], bias=False))
                torch.nn.init.normal_(modules[i].weight)
            else:
                out_linear_dim = input_dim * output_dim * kernel_size[i]
                modules.append(nn.Sequential(nn.Linear(latent_params[-1]['dim'], out_linear_dim), nn.ReLU()))
                torch.nn.init.normal_(modules[-1].weight)

            #torch.nn.init.constant_(modules[i].bias,0)

        self.target_size = checktuple(in_params['dim'])
        if in_params.get('channels') is not None:
            self.target_size = (in_params.get('channels'),)+self.target_size
        self.layers = nn.ModuleList(modules)
        self.nn_lin = module_params.get('nn_lin')
        if latent_params:
            self.forward = self.forward_conditioned
        else:
            self.forward = self.forward_unconditioned


    def forward_unconditioned(self, inp, z=None):
        input_shape = inp.shape
        squeeze = False
        if len(input_shape) != self.conv_dim + 2:
            inp = inp.unsqueeze(1)
            squeeze = True
        for module in self.layers:
            inp = module(inp)
        if self.nn_lin:
            inp = getattr(torch.nn, self.nn_lin)()(inp)
        if squeeze:
            inp = inp.squeeze()
        out = inp.__getitem__((slice(None),)+ tuple([slice(t) for t in self.target_size]))
        return out

    def forward_conditioned(self, inp, z=None):
        input_shape = inp.shape
        kernels = []
        current_out = inp
        squeeze = False
        if len(input_shape) != self.conv_dim + 2:
            inp = inp.unsqueeze(1)
            squeeze = True
        for i, layer in self.layers:
            input_dim = self.input_channels if i==0 else self.channels[i-1]
            output_dim = self.input_channels if i==nlayers else self.channels[i]
            kernel_out = layer(z[1]).view(output_dim, input_dim, self.kernel_size[i])
            current_out = self.conv_fun_hash[self.conv_dim](current_out, kernel_out, bias=False, padding=self.paddings[i])
        if self.nn_lin:
            inp = getattr(torch.nn.functional, self.nn_lin)(inp)
        if squeeze:
            inp = inp.squeeze()
        out = inp.__getitem__((slice(None),)+ tuple([slice(t) for t in self.target_size]))
        return out


class ResidualConvReinforcementModule(ConvReinforcementModule):
    init_value = 1e-3
    def __init__(self, in_params, module_params):
        module_params['nn_lin'] = None
        super(ResidualConvReinforcementModule, self).__init__(in_params, module_params)
        '''
        for i in range(len(self.layers)):
            self.layers[i].weight.data = self.layers[i].weight.data * self.init_value
        '''

    def forward(self, input):
        out = super().forward(input)
        return out + input


######################################################################
# -----    Reinforcement methods based on static loss
######################################################################


class LossReinforcement(Criterion):
    reinforcement_module_hash = {'mlp':MLPReinforcementModule, 'mlp_residual':ResidualMLPReinforcementModule,
                                 'conv':ConvReinforcementModule, 'conv_residual':ResidualConvReinforcementModule}

    def __init__(self, input_params, reinforce_params={"nlayers":1, "nn_lin":None}, latent_params=None,  sample=False, **kwargs):
        super(LossReinforcement, self).__init__()
        self.module_type = reinforce_params.get('type', 'linear')
        self.pinput = input_params
        self.platent = latent_params
        self.lr = reinforce_params.get('lr', 1e-3)
        self.size_average = reinforce_params.get('size_average', False)
        self.tolerance = reinforce_params.get('tolerance', 1.0)
        self.module = self.get_reinforcement_module(input_params, reinforce_params, latent_params=latent_params, **kwargs)
        self.sample = sample
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        self.warmup = reinforce_params.get('warmup', 0)


    def get_reinforcement_module(self, input_params, module_options = {}, module_type=None, **kwargs):
        module_type = module_type or self.module_type
        module =  self.reinforcement_module_hash[module_type](input_params, module_options, **kwargs)
        return module
#        elif module_type == 'conv'
        
    # Module forwards
    def forward(self, vae_out, detach=True, **kwargs):
        if type(self) == LossReinforcement:
            raise NotImplementedError('LossReinforcement is not supposed to be called directly!')
        if self.sample:
            module_input = vae_out['x_params'].rsample()
        else:
            module_input = vae_out['x_params']
            if hasattr(module_input, 'mean'):
                module_input  = module_input.mean
            else:
                raise Exception('No reinforcement heuristic designed for type : %s'%type(module_input))
        z = None
        if self.platent is not None:
            z = vae_out['z_enc']
        if detach:
            module_input = module_input.detach()
            if z:
                z = [z_tmp.detach() for z_tmp in z]
        vae_out['x_reinforced'] = self.module(module_input, z=z)
        return vae_out

    def __call__(self, out, model=None, target=None, sample=False, epoch=None, backward=True, optimize=True, retain_graph=False, *args, **kwargs):
        if not 'x_reinforced' in out.keys():
            out = self.forward(out)
        loss, losses = self.loss(out=out, target=target, sample=sample, epoch=epoch, *args, **kwargs)
        if backward and loss.requires_grad:
            loss.backward(retain_graph=retain_graph)
        if optimize:
            self.step()
        return loss, losses


    # Module loss
    def loss(self, out=None, model=None, target=None, sample=False, epoch=None, *args, **kwargs):
        x_resyn = out.get('x_reinforced')
        if x_resyn is not None:
            losses = (self.get_reinforced_error(x_resyn, target),)
            loss = sum(losses) * 1/(2*(self.tolerance)**2)

            #pdb.set_trace()
            if self.warmup and epoch is not None:
                loss = min(1.0, epoch/self.warmup)*loss
            return loss, (loss.detach().cpu().numpy(),)

    # to be overriden depending of the loss
    def get_reinforced_error(self, x_resyn, x_original):
        pass

    # Optimization functions
    def step(self, retain_graph=False):
        #print_grad_stats(self)
        self.optimizer.step()
        if not retain_graph:
            self.zero_grad()


class L2Reinforcement(LossReinforcement):
    def get_reinforced_error(self, x_resyn, x_original):
        x_original = x_original.to(next(self.parameters()).device).float()
        loss = torch.nn.functional.mse_loss(x_resyn, x_original, size_average = self.size_average)
        if not self.size_average:
            loss /= x_resyn.shape[0]
        return loss

    def get_named_losses(self, losses):
        return {'l2reinforce':losses[0]}

class L1Reinforcement(LossReinforcement):
    def __init__(self, *args, smooth=True, **kwargs):
        super(L1Reinforcement, self).__init__(*args, **kwargs)
        self.smooth = smooth

    def get_reinforced_error(self, x_resyn, x_original):
        x_original = x_original.to(next(self.parameters()).device)
        if self.smooth:
            loss = torch.nn.functional.smooth_l1_loss(x_resyn, x_original, size_average = self.size_average)
        else:
            loss = torch.nn.functional.l1_loss(x_resyn, x_original, size_average = self.size_average)
        if not self.size_average:
            loss /= x_resyn.shape[0]
        return loss

    def get_named_losses(self, losses):
        return {'l1reinforce':losses[0]}

class AdversarialReinforcement(LossReinforcement):
    def __init__(self, pinput, adversarial_params={}, poptim={}, **kwargs):
        super(AdversarialReinforcement, self).__init__(pinput, **kwargs)
        self.adv_loss = Adversarial(pinput, adversarial_params, poptim)

    def get_reinforced_error(self, x_resyn, x_original):
        loss, losses = self.adv_loss(x_resyn, x_original)
        return loss

    def init_optimizer(self, optim_params):
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-5})
        self.optimizer = getattr(torch.optim, alg)(self.parameters(), **optim_args)

        scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
        scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
        self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)(self.optimizer, **scheduler_args)

    def get_named_losses(self, losses):
        return {'adv_reinforce': losses[0]}

