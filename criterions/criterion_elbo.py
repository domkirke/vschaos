import torch, pdb
from . import utils
from ..distributions.distribution_priors import get_default_distribution
from . import reduce, Criterion, CriterionContainer, KLD, LogDensity, NaNError, MultiDivergence
from ..utils import checklist, apply, print_stats


def scale_prior(prior, size):
    #TODO has to be implemented to prevent unscaled priors for KLD
    return prior
def regularize_logdets(logdets):
    if logdets is None:
        return 0
    if len(logdets[0].shape) == 1:
        logdets = torch.cat([l.unsqueeze(1) for l in logdets], dim = 1)
    elif len(logdets[0].shape)>=2:
        logdets = torch.cat(logdets, dim=-1)
    return logdets


class ELBO(CriterionContainer):
    reconstruction_class = LogDensity
    regularization_class = KLD

    def __init__(self, warmup=100, beta=1.0, warmup_exp=1, reconstruction_loss=None, regularization_loss=None, *args, **kwargs):
        # losses
        reconstruction_loss = reconstruction_loss if reconstruction_loss is not None else self.reconstruction_class
        regularization_loss = regularization_loss if regularization_loss is not None else self.regularization_class
        if issubclass(type(reconstruction_loss), type):
            reconstruction_loss = reconstruction_loss(*args, **kwargs)
        if issubclass(type(regularization_loss), type):
            regularization_loss = regularization_loss(*args, **kwargs)
        super(ELBO, self).__init__(criterions=[reconstruction_loss, regularization_loss])
        self.reconstruction_loss = reconstruction_loss
        self.regularization_loss = regularization_loss
        # parameters
        self.warmup = warmup
        self.beta = beta
        self.warmup_exp = max(warmup_exp,0)


    def get_warmup_factors(self, latent_params, epoch=None, beta=None, warmup=None):
        # get warmup & beta sub-parameters
        warmup = warmup or self.warmup
        if issubclass(type(warmup), list):
            assert len(warmup) >= len(latent_params)
        else:
            warmup = [warmup]*len(latent_params)
        beta = beta or self.beta
        if issubclass(type(beta), list):
            assert len(beta) >= len(latent_params)
        else:
            beta = [beta]*len(latent_params)
        scaled_factors = [min(((epoch+1)/(warmup[i]-1))**self.warmup_exp, 1.0)*beta[i] if warmup[i] != 0 and epoch is not None else beta[i] for i in range(len(latent_params))]
        return scaled_factors

    def get_reconstruction_params(self, model, out, target, epoch=None, callback=None):
        callback = callback or self.reconstruction_loss
        input_params = checklist(model.pinput); target = checklist(target)
        rec_params = []
        x_params = checklist(out['x_params'])
        for i, ip in enumerate(input_params):
            rec_params.append((callback, {'x_params': x_params[i], 'target': model.format_input_data(target[i]), 'input_params': ip, 'epoch':epoch}, 1.0))
        return rec_params

    def get_regularization_params(self, model, out, epoch=None, beta=None, warmup=None, callback=None):

        def parse_layer(latent_params, out, layer_index=0):
            if issubclass(type(latent_params), list):
                return [parse_layer(latent_params[i], utils.get_latent_out(out,i)) for i in range(len(latent_params))]
            #TODO if not z_params_enc, make montecarlo estimation
            params1 = out["z_params_enc"];
            if params1.requires_preflow:
                out1 = out['z_preflow_enc']
            else:
                out1 = out['z_enc']
            # decoder parameters
            prior = latent_params.get('prior') or None
            if prior is not None:
                params2 = scale_prior(prior, out['z_enc'])
            elif out.get('z_params_dec') is not None:
                params2 = out['z_params_dec']
                out2 = out["z_dec"]
            else:
                params2 = get_default_distribution(latent_params['dist'], out['z_params_enc'].batch_shape,
                                                   device=out['z_enc'].device)
                out2 = params2.rsample()

            #pdb.set_trace()
            return {"params1":params1, "params2":params2, "out1":out1, "out2":out2}

        # retrieve regularization parameters
        latent_params = checklist(model.platent)
        beta = beta or self.beta; beta = checklist(beta, n=len(latent_params))
        warmup = warmup or self.warmup
        factors = self.get_warmup_factors(latent_params, epoch=epoch, beta=beta, warmup=warmup)

        # parse layers
        reg_params = []; regularization_loss = checklist(self.regularization_loss, n=len(latent_params))
        for i, ip in enumerate(latent_params):
            parsed_layer = parse_layer(ip, utils.get_latent_out(out, i))
            if issubclass(type(ip), list):
                reg_params.extend([(regularization_loss[i], p, factors[i]) for p in parsed_layer])
            else:
                reg_params.append((regularization_loss[i], parsed_layer, factors[i]))
        return reg_params

    def loss(self, model = None, out = None, target = None, epoch = None, beta=None, warmup=None, *args, **kwargs):
        assert not model is None and not out is None and not target is None, "ELBO loss needs a model, an output and an input"
        # parse loss arguments
        reconstruction_params = self.get_reconstruction_params(model, out, target, epoch=epoch)
        beta = beta or self.beta
        regularization_params = self.get_regularization_params(model, out, epoch=epoch, beta=beta, warmup=warmup)
        logdets = tuple()
        # get warmup coefficient

        full_loss = 0; rec_errors=tuple(); reg_errors=tuple()
        # get reconstruction error
        for i, rec_args in enumerate(reconstruction_params):
            #reduction = 'seq' if model.take_sequences else 'none'
            rec_loss, rec_losses = rec_args[0](**rec_args[1], reduction=self.reduction, is_sequence=model.take_sequences,**kwargs)
            full_loss = full_loss + rec_args[2]*rec_loss if rec_args[2] != 0. else full_loss
            rec_errors = rec_errors + rec_losses

         # get latent regularizoation error
        for i, reg_args in enumerate(regularization_params):
            reg_loss, reg_losses = reg_args[0](**reg_args[1], reduction=self.reduction, is_sequence=model.take_sequences, **kwargs)
            #print(reg_loss, reg_args[2])
            full_loss = full_loss + reg_args[2]*reg_loss if reg_args[2] != 0. else full_loss
            reg_errors = reg_errors + (reg_losses,)
            if out.get('logdets') is not None:
                if out['logdets'][i] is None:
                    continue
                logdet_error = regularize_logdets(out['logdets'][i]) 
                if torch.is_tensor(logdet_error):
                    logdet_error = reduce(logdet_error, self.reduction)
                full_loss = full_loss - logdet_error
                logdets = logdets + (float(logdet_error),)

        return full_loss, (rec_errors, reg_errors, *logdets)
             
    def get_named_losses(self, losses):
        named_losses = {}
        if issubclass(type(self.reconstruction_loss), list):
            rec_losses = [self.reconstruction_loss[i].get_named_losses(losses[0][i]) for i in range(len(self.reconstruction_loss))]
            for r in rec_losses:
                named_losses = {**r, **named_losses}
        else:
            named_losses = {**self.reconstruction_loss.get_named_losses(losses[0]), **named_losses}

        if issubclass(type(self.regularization_loss), MultiDivergence):
            named_losses = {**self.regularization_loss.get_named_losses(losses[1]), **named_losses}
        elif issubclass(type(self.regularization_loss), list):
            reg_losses = [self.regularization_loss[i].get_named_losses(losses[1][i]) for i in range(len(self.regularization_loss))]
            for r in reg_losses:
                named_losses = {**r, **named_losses}
        else:
            named_losses = {**named_losses, **self.regularization_loss.get_named_losses(losses[1])}
        return named_losses
                

