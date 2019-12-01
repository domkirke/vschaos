#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:45:41 2018

@author: chemla
"""
from ..criterions.criterion_criterion import reduce
import torch.distributions.kl as kl
import torch.nn.functional as func
import pdb
from .. import utils

from .criterion_elbo import ELBO, KLD, LogDensity
from .criterion_functional import MSE


class SCANLoss(ELBO):
    transfer_loss = KLD
    reconstruction_loss = [LogDensity, LogDensity]
    regularization_loss = KLD
    def __init__(self, warmup=100, beta=1.0, distance='kld', cross_factors=[10.0, 0.0], *args, **kwargs):
        super(SCANLoss, self).__init__(warmup=warmup, beta=beta, *args, **kwargs)
        self.distance_type = distance
        self.cross_factors = cross_factors
        if distance == "mse":
            self.transfer_callbacks = MSE

    def get_reconstruction_params(self, model, out, target):
        rec_params = []
        for i in range(len(model)):
            current_model = model[i]; current_out = out[i]; current_target = target[i]
            if not issubclass(type(model[i]), list):
                current_model = [current_model]; current_out = [current_out]; current_target = [current_target]
            for j in range(len(current_model)):
                rec_params.extend(super(SCANLoss, self).get_reconstruction_params(current_model[j], current_out[j], current_target[j], callback=self.reconstruction_loss[i]))
        return rec_params

    def get_regularization_params(self, model, out, beta=None, *args, **kwargs):
        reg_params = []
        for i in range(len(model)):
            current_model = model[i]; current_out = out[i]; current_beta = beta[i]
            if not issubclass(type(model[i]), list):
                current_model = [current_model]; current_out = [current_out];
            for j in range(len(current_model)):
                reg_params.extend(super(SCANLoss, self).get_regularization_params(current_model[j], current_out[j], beta=current_beta, *args, **kwargs))

        return reg_params
        # beta = beta or self.beta
        # if not issubclass(type(beta), list):
        #     beta = [beta]*len(model)

    def get_transfer_params(self, outs, cross_factors=None, distance_type=None, layer=-1, *args, **kwargs):
        assert len(outs) == 2, "so far only 2-distributions matching is handled"
        transfer_latent_params = [utils.get_latent_out(outs[i], layer) for i in range(len(outs))]
        transfer_args = []
        cross_factors = cross_factors or self.cross_factors
        distance_type = distance_type or self.distance_type
        if distance_type == "kld":
            for i in range(len(transfer_latent_params)):
                if not issubclass(type(transfer_latent_params[i]['z_params_enc']), list):
                    transfer_latent_params[i] = [transfer_latent_params[i]]
            assert len(transfer_latent_params[0]) == len(transfer_latent_params[1])
            for i in range(len(transfer_latent_params[0])):
                transfer_args.append((self.transfer_loss, {'params1':transfer_latent_params[0][i]["z_params_enc"],
                                      'params2':transfer_latent_params[1][i]["z_params_enc"]}, cross_factors[0]))
                transfer_args.append((self.transfer_loss, {'params1':transfer_latent_params[1][i]["z_params_enc"],
                                      'params2':transfer_latent_params[0][i]["z_params_enc"]}, cross_factors[1]))

        return transfer_args

    def loss(self, model=None, out=None, target=None, epoch=None, write=None, *args, **kwargs):
        # ELBO error
        full_loss, full_losses = super(SCANLoss, self).loss(model, out, target, epoch=epoch)

        # transfer errors
        distance_type = kwargs.get('distance', self.distance_type)
        cross_factors = kwargs.get('betas', self.cross_factors)
        transfer_params = self.get_transfer_params(out, cross_factors, distance_type)
        transfer_losses = tuple()
        for i, rec_args in enumerate(transfer_params):
            tr_loss, tr_losses = rec_args[0](*args, **kwargs)(**rec_args[1])
            full_loss = full_loss + rec_args[2]*tr_loss
            transfer_losses = transfer_losses + tr_losses
        full_losses = full_losses + transfer_losses

        return full_loss, full_losses

    def get_named_losses(self, losses):
        dict_losses = {'rec_losses':losses[0], 'reg_losses':losses[1], 'transfer_losses':losses[2]}
        return dict_losses


class MixtureSCANLoss(SCANLoss):
    reconstruction_loss = [MSE, LogDensity]

    def __init__(self, alpha=0.0, gamma=1.0, delta=0.0, zeta=0.0, *args, **kwargs):
        super(MixtureSCANLoss, self).__init__(*args, **kwargs)
        self.alpha = alpha     # mixture reconstruction loss
        self.gamma = gamma     # reconstruction of solo signals
        self.delta = delta     # latent KLD supervision coefficient
        self.zeta = zeta       # mixture coefficient supervision

    def get_transfer_params(self, outs, cross_factors=None, distance_type=None, layer=-1, *args, **kwargs):
        outs_sig = utils.get_latent_out(outs[0], layer)
        outs_symb = {}
        if not issubclass(type(layer), list):
            layer = [layer]*len(outs[1])
        current_out = [utils.get_latent_out(outs[1][i], layer[i]) for i in range(len(outs[1]))]
        for k in current_out[0].keys():
                outs_symb[k] = [current_out[i][k] for i in range(len(current_out))]
        return super(MixtureSCANLoss, self).get_transfer_params([outs_sig, outs_symb], cross_factors=cross_factors,
                                                                distance_type=distance_type, layer=0, *args, **kwargs)

    def loss(self, models=None, outs=None, target=None, epoch=None, solos=None, solo_models=None, random_weights=None, write=None, *args, **kwargs):
        # assert some arguments
        if self.delta != 0:
            assert solo_models, "needs independent VAEs since delta != 0"
        if self.gamma != 0:
            assert solos, "needs independent signals since gamma != 0"

        # SCAN loss
        full_loss, full_losses = super(MixtureSCANLoss, self).loss(models, outs, target, epoch, write=False, *args, **kwargs)

        # solo reconstruction error
        solo_rec_errors = []; kld_enc_losses = []
        for i in range(len(solos)):
            # error between solo reconstruction and original solo distribution
            if self.gamma != 0:
                solo_rec_errors.append(LogDensity(reduction=self.reduction)(x_params=outs[0]['x_solo_params'][i],
                                                                       target=models[0].format_input_data(solos[i]),
                                                                       input_params=models[0].pinput)[0])
            # latent KLD supervision
            if self.delta != 0:
               current_solo = solo_models[i].format_input_data(solos[i], requires_grad=True)
               latent_dist = solo_models[i].encode(current_solo)[-1]['out_params']
               if self.delta > 0:
                    kld_enc_losses.append(reduce(kl.kl_divergence(outs[0]['z_params_enc'][-1][i], latent_dist), reduction=self.reduction))
               else:
                    kld_enc_losses.append(reduce(kl.kl_divergence(latent_dist, outs[0]['z_params_enc'][-1][i]), reduction=self.reduction))

        full_loss = full_loss + self.gamma*sum(solo_rec_errors) + abs(self.delta)*sum(kld_enc_losses)
        full_losses = full_losses + (tuple(solo_rec_errors),) + (tuple(kld_enc_losses),)

        # mixture coefficient learning
        if self.zeta != 0:
            raise NotImplementedError() #TODO

        if write:
            self.write(write, full_losses)
        return full_loss, full_losses


    def get_named_losses(self, losses):
        dict_losses = {'reconstruction':losses[0], 'regularization':losses[1],
                       'transfer_1':losses[2], 'transfer_2':losses[3],
                       'solo_rec':losses[4], 'kld_supervision':losses[5]}
        return dict_losses

