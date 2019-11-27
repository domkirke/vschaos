from ..modules.modules_recurrent import *
from math import ceil
import torch.nn as nn
import pdb, dill, numpy as np
from .vae_vanillaVAE import VanillaVAE
from .vae_recurrentVAE import VRNN, ShrubVAE
from ..modules.modules_prediction import FlowPrediction
#TODO Weiner process
from ..utils import denest_dict, flatten_seq_method, GPULogger, merge_dicts, checklist
from ..utils.misc import concat_distrib, print_stats, crossed_select, dist_crossed_select

logger = GPULogger(verbose=False)

# As here Prediction VAEs are meta-classes that combine a VAE and a predictior, all the methods
#   are defined separately from the object and dynamically added to a given class

def predictive_vae_init(self, *args, **kwargs):
    self.vae_class.__init__(self, *args, **kwargs)
    self.prediction_class.__init__(self, *args, **kwargs)


def decode_prediction(self, z_predicted, *args, **kwargs):
    return self.vae_class.decode(self, z_predicted)


def concat_prediction(self, vae_out, prediction_out, n_preds=None, teacher_prob=None, epoch=None, **kwargs):
    z_predicted = prediction_out.get('out')
    z_params_predicted = prediction_out.get('out_params')
    teacher_prob = teacher_prob or self.teacher_prob or 0
    if self.teacher_warmup != 0 and epoch is not None:
        teacher_prob = (1 - min(1.0, epoch / self.teacher_warmup)*(1-teacher_prob))
    # 1 means taking true encoded position
    #print('teacher_prob', teacher_prob)
    mask = torch.distributions.Bernoulli(teacher_prob).sample(sample_shape=[z_predicted.shape[0]]).to(device=z_predicted.device)
    #1print(mask)
    if torch.sum(mask) < z_predicted.shape[0]:
        x_decoded = self.decode_prediction(z_predicted, target_seq=n_preds)
        seq_length = z_predicted.shape[1]
        try:
            vae_out['z_params_enc'][-1] = concat_distrib([vae_out['z_params_enc'][-1][:, :-seq_length],
                dist_crossed_select(mask, vae_out['z_params_enc'][-1][:, -seq_length:], z_params_predicted)], dim=1, unsqueeze=False)
        except NotImplementedError:
            pass
        vae_out['z_enc'][-1] = torch.cat([vae_out['z_enc'][-1][:, :-seq_length],
                crossed_select(mask, vae_out['z_enc'][-1][:, -seq_length:], z_predicted)], dim=1)
        for layer in range(len(x_decoded)-1):
            seq_length = x_decoded[layer+1]['out'].shape[1]
            vae_out['z_params_dec'][layer] = concat_distrib([vae_out['z_params_dec'][layer][:, :-seq_length],
                dist_crossed_select(mask, vae_out['z_params_dec'][layer][:, -seq_length:], x_decoded[layer+1]['out_params'])], dim=1, unsqueeze=False)
            vae_out['z_dec'][layer] = torch.cat([vae_out['z_dec'][layer][:, :-seq_length],
                crossed_select(mask, vae_out['z_dec'][layer][:, -seq_length:], x_decoded[layer+1]['out'])], dim=1)
            #TODO flows?
        vae_out['x_params'] = concat_distrib([vae_out['x_params'][:, :-n_preds], 
            dist_crossed_select(mask,  x_decoded[0]['out_params'], vae_out['x_params'][:, -n_preds:])], dim=1, unsqueeze=False)

    if prediction_out.get('logdets'):
        logdets = vae_out.get('logdets', [None]*len(self.platent)) or [None]*len(self.platent)
        if logdets[-1] is None:
            logdets[-1] = prediction_out['logdets']
        else:
            logdets[-1].append(prediction_out['logdets'])
        vae_out['logdets'] = logdets

    vae_out['prediction'] = prediction_out
    return vae_out


def forward_and_predict(self, x, *args, n_preds=None, predict=True, **kwargs):
    n_preds = n_preds or self.n_predictions
    if n_preds >= (x.shape[1] - 1):
        raise Exception('cannot prediction %d items for sequence of size %d'%(n_preds, x.shape[1]))
    logger('strat forward')
    #pdb.set_trace()
    # forward in vae
    vae_out = self.vae_class.forward(self, x, *args, **kwargs)
    # forward in predictor
    if predict:
        logger('start prediction')
        prediction_out = self.prediction_class.forward(self, vae_out, **kwargs)
        # mixed forward
        logger('start concatenation')
        vae_out = self.concat_prediction(vae_out, prediction_out, n_preds=n_preds, **kwargs)
    logger('return')
    return vae_out


def encode_and_predict(self, x, *args, predict=False, **kwargs):
    vae_out = self.vae_class.encode(self, x, **kwargs)
    if predict:
        formatted_out = {'z_params_enc':[vae_out[i].get('out_params') for i in range(len(vae_out))],
                         'z_enc':[vae_out[i].get('out') for i in range(len(vae_out))]}
        prediction_out = self.prediction_class.forward(formatted_out)
        return vae_out, prediction_out
    else:
        return vae_out

def decode_and_predict(self, x, *args, n_preds=None, predict=False, inplace=False, context=5, **kwargs):
    if predict:
        x_formatted={'z_enc':x}
        prediction_out = self.prediction_class.forward(self, x_formatted)
        if inplace:
            n_preds = n_preds or prediction_out['out'].shape[1]
            x[-1][:, -n_preds:] = prediction_out['out'][:, :n_preds]
        else:
            x[-1] = torch.cat([x[-1], prediction_out['out']], dim=1)
        return self.vae_class.decode(self, x, *args, **kwargs)
    else:
        return self.vae_class.decode(self, x, *args, **kwargs)


def init_optimizer_predictive(self, optim_params):
    optim_mode = optim_params.get('predictor_mode', 'joint')
    optim_predictor_params = optim_params.get('predictor', optim_params) or optim_params
    #self.vae_class.init_optimizer(self, optim_params, init_scheduler= (optim_mode == 'joint'))
    self.vae_class.init_optimizer(self, optim_params, init_scheduler=True)

    vae_optimizer = None
    if optim_mode == "joint":
        vae_optimizer = self.optimizers.get('default')
    self.prediction_class.init_optimizer(self, optim_predictor_params, optimizer=vae_optimizer)
    if optim_mode == "joint":
        self.vae_class.init_scheduler(self, optim_params)


def optimize_prediction(self, loss, *args, **kwargs):
    optim_mode = self.optim_params.get('predictor_mode', 'joint')
    if optim_mode != "fixed":
        self.vae_class.optimize(self, loss, *args, **kwargs)
    if optim_mode != "joint":
        self.prediction_class.optimize(self, loss, *args, **kwargs)

def schedule_prediction(self, loss, *args, **kwargs):
    self.vae_class.schedule(self, loss, *args, **kwargs)
    self.prediction_class.schedule(self, loss, *args, **kwargs)


def prediction_reduce(cls):
    print(cls.vae_class.__reduce__())
    print(cls.prediction_class.__reduce__())
    return ""

def prediction_get_dict(self, *args, **kwargs):
    vae_dict = self.vae_class.get_dict(self, **kwargs)
    return vae_dict
    # prediction_dict = self.prediction_class.get_dict(self, **kwargs)
    # return {'vae':vae_dict, 'prediction':prediction_dict}


def get_prediction_vae(vae_class, prediction_class, name=None):
    class_name = name or '%s%s'%(vae_class.__name__, prediction_class.__name__)
    new_class = type(class_name, (vae_class, prediction_class), {})
    new_class.vae_class = vae_class
    new_class.prediction_class = prediction_class

    new_class.__init__ = predictive_vae_init
    new_class.__repr__ = vae_class.__repr__
    new_class.__getstate__ = prediction_reduce
    new_class.__reduce__ = prediction_reduce
    new_class.forward = forward_and_predict
    new_class.encode = encode_and_predict
    new_class.decode = decode_and_predict
    new_class.concat_prediction = concat_prediction
    new_class.init_optimizer = init_optimizer_predictive
    new_class.optimize = optimize_prediction
    new_class.schedule = schedule_prediction
    new_class.take_sequences = True
    new_class.cuda = nn.Module.cuda
    new_class.decode_prediction = decode_prediction
    new_class.get_dict = prediction_get_dict
    return new_class


### Prediction Module definition
class PredictionModule(nn.Module):
    PredictionClass = FlowPrediction
    take_sequences = True
    requires_recurrent = False
    def __init__(self, *args, prediction_params=None, **kwargs):
        self.init_predictor(prediction_params)
        self.prediction_params = prediction_params
        self.n_predictions = prediction_params['n_predictions']
        self.teacher_prob = prediction_params.get('teacher_prob', 0)
        self.teacher_warmup = prediction_params.get('teacher_warmup', 0)

    @property
    def encode_predictions(self):
        if self.prediction_module is None:
            return None
        else:
            try:
                return self.prediction_module.encode_predictions
            except AttributeError:
                pass
            return None

    @encode_predictions.setter
    def encode_predictions(self, *args):
        return AttributeError('encode_predictions cannot be affected to another value')

    def init_predictor(self, prediction_params):
        prediction_class = prediction_params.get('class', self.PredictionClass)
        precurrent = None
        if hasattr(self, 'precurrent'):
            precurrent= self.precurrent[-1]
        self.prediction_module = prediction_class(self.platent[-1], prediction_params,
                                                  recurrent_params = precurrent, hidden_params=self.phidden[-1])
        if hasattr(self.prediction_module, 'requires_recurrent'):
            self.requires_recurrent = self.prediction_module.requires_recurrent

    def init_optimizer(self, optim_params, optimizer = None):
        if optimizer is not None:
            optimizer.add_param_group({'params':self.prediction_module.parameters()})
        else:
            alg = optim_params.get('optimizer', 'Adam')
            optim_args = optim_params.get('optim_args', {'lr':1e-3})
            optimizer = getattr(torch.optim, alg)([{'params':self.parameters()}], **optim_args)
            if not hasattr(self, 'optimizers'):
                self.optimizers = {}
            self.optimizers['predictor'] = optimizer

            scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
            scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
            self.schedulers['predictor'] = getattr(torch.optim.lr_scheduler, scheduler)(self.optimizers['predictor'], **scheduler_args)

    def optimize(self, loss, options={}, retain_graph=False, *args, **kwargs):
        if 'predictor' in self.optimizers:
            self.optimizers['predictor'].step()

    def schedule(self, loss, options={}):
        if 'predictor' in self.schedulers:
            self.schedulers['predictor'].step(loss)

    def forward(self, out, **kwargs):
        if not self.encode_predictions:
            raise NotImplemented
        pred_out = self.prediction_module(out, **kwargs)
        return pred_out


PredictiveVAE = get_prediction_vae(VanillaVAE, PredictionModule, 'PredictiveVAE')
PredictiveVRNN = get_prediction_vae(VRNN, PredictionModule, 'PredictiveVRNN')


def decode_prediction_shrub(self, z_predicted, target_seq=None):
    return self.vae_class.decode(self, z_predicted, target_seq=target_seq)


def predictive_shrub_init(self, input_params, latent_params, hidden_params=None, *args, prediction_params=None, **kwargs):
    n_predictions = prediction_params['n_predictions']
    shrub_dims = [h.get('path_length') for h in hidden_params[1:]]
    n_predictions_shrub = n_predictions
    for shrub_dim in shrub_dims:
        if shrub_dim is None:
            n_predictions_shrub = 1
        else:
            #n_predictions_shrub = n_predictions // np.cumprod(shrub_dims[:-1])[-1]
            n_predictions_shrub = ceil(n_predictions / np.cumprod(shrub_dims)[-1])
    prediction_params['n_predictors'] = n_predictions_shrub
    self.vae_class.__init__(self, input_params, latent_params, hidden_params, *args, prediction_params=prediction_params,**kwargs)
    self.prediction_class.__init__(self, input_params, latent_params, hidden_params, *args, prediction_params=prediction_params,**kwargs)

def encoder_and_predict_shrub(self, x, *args, predict=False, return_shifts=True, **kwargs):
    vae_out = self.vae_class.encode(self, x, **kwargs)
    shifts = None
    if type(vae_out) == tuple:
        vae_out, shifts = vae_out
    if predict:
        formatted_out = {'z_params_enc':[vae_out[i].get('out_params') for i in range(len(vae_out))],
                         'z_enc':[vae_out[i].get('out') for i in range(len(vae_out))]}
        if vae_out[-1].get('hidden') is not None:
            formatted_out['recurrent_out'] = vae_out[-1]['hidden']
        prediction_out = self.prediction_class.forward(self, formatted_out)
        if return_shifts:
            return vae_out, shifts, prediction_out
        else:
            return vae_out, prediction_out

    else:
        if return_shifts:
            return vae_out, shifts
        else:
            return vae_out

PredictiveShrubVAE = get_prediction_vae(ShrubVAE, PredictionModule, 'PredictiveShrubVAE')
PredictiveShrubVAE.encode = encoder_and_predict_shrub
PredictiveShrubVAE.decode_prediction = decode_prediction_shrub
PredictiveShrubVAE.__init__ = predictive_shrub_init
