from ..modules.modules_recurrent import *
from numpy import ceil
from .vae_vanillaVAE import VanillaVAE
#TODO Weiner process
from .. import distributions as dist
from ..distributions.distribution_priors import IsotropicGaussian
from ..utils import denest_dict, checklist, pack_output, flatten_seq, reshape_distribution, view_distribution, cat_distribution, checklist, apply_method
from ..utils import crossed_select, dist_crossed_select, print_stats, GPULogger
from functools import reduce
import gc, math


class RVAE(VanillaVAE):
    HiddenModuleClass = [RVAEEncoder, RVAEDecoder]
    take_sequences = True

    def forward(self, x, *args, **kwargs):
        n_steps = x.shape[1]
        return super(RVAE, self).forward(x, *args, n_steps=n_steps, **kwargs)

    # def format_input_data(self, x, requires_grad=True, pinput=None, onehot=True, sample_norm=False, *args, **kwargs):
    #     super(RVAE, self).format_input_data(x, requires_grad, pinput, onehot, sample_norm)


class VRNNRecurrentModule(RecurrentModule):
    def __setattr__(self, name, value, register_parameters=True):
        if register_parameters:
            torch.nn.Module.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def __init__(self, pins=None, phidden=None, platent=None, *args, embedding=None, **kwargs):
        assert pins and platent and phidden
        self.pins = pins; self.phidden = phidden; self.platent = platent
        super().__init__([pins, platent], phidden, *args, **kwargs)
        if embedding is not None:
            self.set_embedding(embedding)
        self.flatten = True

    def set_embedding(self, embedding):
        self.pins = embedding.pins
        self.__setattr__('embedding',  embedding, register_parameters=False)

    def forward(self, x, y=None, sample=True, return_states=False, bypass_embedding=False, *args, **kwargs):
        x, z = tuple(x)
        if self.embedding and not bypass_embedding:
            x = self.embedding(x)
        return super(VRNNRecurrentModule, self).forward([x,z], y=y, *args, **kwargs)


class VRNN(VanillaVAE):
    HiddenModuleClass = [VRNNEncoder, VRNNDecoder]
    RecurrentClass = VRNNRecurrentModule
    PriorClass = HiddenModule
    take_sequences = True

    def init_modules(self, input_params, latent_params, hidden_params, recurrent_params=None, prior_params=None, *args, **kwargs):
        self.precurrent = checklist(recurrent_params)
        self.pprior = prior_params
        assert recurrent_params
        self.init_recurrent_module(self.precurrent, hidden_params[0].get('encoder', hidden_params[0]), latent_params[0])
        self.init_prior_module(self.precurrent, latent_params, prior_params)
        super(VRNN, self).init_modules(input_params, latent_params, hidden_params,
                                       recurrent_module=self.recurrent_module, prior_module=self.prior_module, **kwargs)
        if self.precurrent[0].get('take_encoder_embedding', True):
            self.recurrent_module.set_embedding(self.encoders[0].hidden_modules)

    def init_recurrent_module(self, recurrent_params, input_params, latent_params):
        recurrent_params = checklist(recurrent_params)
        recurrent_class = recurrent_params[0].get('class', self.RecurrentClass)
        self.recurrent_module = recurrent_class(pins=input_params, platent=latent_params, phidden=recurrent_params[0])

    def init_prior_module(self, recurrent_params, latent_params, prior_params=None):
        prior_params = prior_params or {'dim':recurrent_params[0]['dim'], 'nlayers':1, 'ignore_flows':1}
        out_params = dict(latent_params[0])
        if prior_params.get('ignore_flows', 1) and out_params.get('flows') is not None:
            del out_params['flows']
        self.prior_module = self.PriorClass(recurrent_params, phidden=prior_params, pouts=latent_params[0])

    def init_optimizer(self, optim_params, init_scheduler=True):
        super(VRNN, self).init_optimizer(optim_params)
        self.optimizers['default'].add_param_group({'params':self.recurrent_module.parameters()})
        self.optimizers['default'].add_param_group({'params':self.prior_module.parameters()})
        super(VRNN, self).init_scheduler(optim_params)

    def encode(self, x, *args, clear=True, **kwargs):
        if clear:
            self.recurrent_module.clear()
        return super(VRNN, self).encode(x, *args, **kwargs)

    def decode(self, z, *args, clear=True, **kwargs):
        if clear:
            self.recurrent_module.clear()
        out = super(VRNN, self).decode(z, *args, **kwargs)
        priors = self.get_prior_distributions(out[0]['recurrent'])
        out.append(priors)
        return out

    def get_prior_init(self, batch_shape):
        device = next(self.parameters()).device
        prior = IsotropicGaussian((batch_shape, self.platent[0]['dim']), device=device)
        return {'out_params':prior, 'out':prior.rsample()}

    def get_prior_distributions(self, previous_h):
        n_seq = previous_h.shape[1]
        prior_dists = [self.get_prior_init(previous_h.shape[0])]
        for i in range(1, n_seq):
            prior_dists.append(self.prior_module(previous_h[:, i-1]))
        prior_dists = utils.merge_dicts(prior_dists, unsqueeze=1, dim=1)
        return prior_dists


    def forward(self, x, y=None, options={}, clear=True, *args, **kwargs):
        if clear:
            self.recurrent_module.clear()
        x = self.format_input_data(x, requires_grad=False)
        enc_out = self.encode(x, y=y, *args, **kwargs)
        dec_out = self.decode(enc_out[-1]['out'], y=y, previous_h = enc_out[0]['recurrent'], *args, **kwargs)

        # if some z were predicted
        if dec_out[0].get('z_pred') is not None:
            dec_out[1]['out_params'] = utils.misc.concat_distrib([dec_out[1]['out_params'], dec_out[0]['z_pred']['out_params']],dim=1, unsqueeze=False)
            dec_out[1]['out'] = torch.cat([dec_out[1]['out'], dec_out[0]['z_pred']['out']], dim=1)

        x_params = dec_out[0]['out_params']
        dec_out = denest_dict(dec_out[1:]) if len(dec_out) > 1 else {}
        enc_out = denest_dict(enc_out)
        return {'x_params':x_params, "logdets":enc_out.get('logdet'), "z_preflow":enc_out.get('out_preflow'),
                'z_params_dec':dec_out.get('out_params'), 'z_dec':dec_out.get('out'),
                'z_params_enc':enc_out['out_params'], 'z_enc':enc_out['out']}



class ShrubVAE(VanillaVAE):
    HiddenModuleClass = [RVAEEncoder, RVAEDecoder]
    take_sequences=True
    loaded_encoder=False
    loaded_decoder=False

    def init_modules(self, input_params, latent_params, hidden_params, from_vae=None, *args, **kwargs):
        hidden_params = checklist(hidden_params)
        encoder = None; decoder = None
        self.teacher_prob = kwargs.get('teacher_prob', 0.) # 0 means only decoder's zs, 1 means only encoders' zs (if available)
        self.teacher_warmup = kwargs.get('teacher_warmup', 0)
        super(ShrubVAE, self).init_modules(input_params, latent_params, hidden_params,
                             encoder = encoder, decoder = decoder, *args, **kwargs)
        if hidden_params[0].get('load'):
            loaded_data = torch.load(hidden_params[0]['load'], map_location="cpu")
            vae = loaded_data['class'].load(loaded_data)
            #if hidden_params[0].get('load') in ["encoder", "full"]:
            self.encoders[0] = vae.encoders[0]; self.loaded_encoder=True
            #elif hidden_params[0].get('load') == ["decoder", "full"]:
            self.decoders[0] = vae.decoders[0]; self.loaded_decoder = True
        self.precurrent = kwargs.get('recurrent_params')
        if not hasattr(self, 'requires_recurrent'):
            self.requires_recurrent = False

    def make_encoders(self, input_params, latent_params, hidden_params, *args, recurrent_params=None, encoder=None, decoder=None, **kwargs):
        encoders = nn.ModuleList()
        assert recurrent_params
        precurrent = checklist(recurrent_params, len(latent_params))

        for layer in range(len(latent_params)):
            if layer==0:
                if encoder is not None:
                    encoders.append(encoder)
                else:
                    encoders.append(self.make_encoder(input_params, latent_params[0], hidden_params[0],
                                                  module_class=HiddenModule,
                                                  name="vae_encoder_%d"%layer,
                                                  *args, **kwargs))
            else:
                encoders.append(self.make_encoder(latent_params[layer-1], latent_params[layer], hidden_params[layer],
                                                  name="vae_encoder_%d"%layer,
                                                  recurrent_params=precurrent[layer-1],
                                                  *args, **kwargs))
        return encoders

    def make_decoders(self, input_params, latent_params, hidden_params, recurrent_params, *args, encoder=None, decoder=None, **kwargs):
        decoders = nn.ModuleList()
        assert recurrent_params
        precurrent = checklist(recurrent_params, len(latent_params))
        for layer in range(len(latent_params)):
            if layer==0:
                if decoder is not None:
                    new_decoder = decoder
                else:
                    new_decoder = VanillaVAE.make_decoder(input_params, latent_params[0], hidden_params[0],
                                                      module_class=HiddenModule,
                                                      name="vae_decoder_%d"%layer,
                                                      encoder = self.encoders[layer-1], *args, **kwargs)
            else:
                new_decoder = self.make_decoder(latent_params[layer-1], latent_params[layer], hidden_params[layer],
                                                name="vae_decoder_%d"%layer,
                                                recurrent_params = precurrent[layer-1],
                                                encoder=self.encoders[layer], *args, **kwargs)
            decoders.append(new_decoder)
        return decoders


    def encode(self, x, y=None, sample=True, from_layer=0, clear=True, return_shifts=True, *args, **kwargs):
        # get through first layer (one z per step)
        outs = []; shifts = [None]*len(self.platent)
        n_batches = x.shape[0]; n_seq = x.shape[1]
        # flatten if first encoder dont take sequences
        if not self.encoders[0].take_sequences:
            x = x.contiguous().view((n_batches*n_seq, *x.shape[2:]))
        # forward first layer
        current_out = self.encoders[0](x, y=y)
        # unflatten in case
        if not self.encoders[0].take_sequences:
            current_out['out_params'] = reshape_distribution(current_out['out_params'], (n_batches, n_seq, self.platent[0]['dim']))
            if current_out.get('out') is None:
                current_out['out'] = apply_method(current_out['out_params'], 'rsample')
        outs.append(current_out)

        # window paths in case
        for layer in range(1, len(self.platent)):
            current_z = current_out['out']
            # old code for path slicing
            '''
            if self.phidden[layer].get('path_length'):
                path_length = self.phidden[layer]['path_length']
                path_overlap = self.phidden[layer].get('path_overlap')
                path_overlap = path_length if not path_overlap else path_overlap
                current_z_sliced =[current_z[:, i*path_overlap:(i*path_overlap+path_length)] for i in range((current_z.shape[1] - path_length)//path_overlap + 1)]
                if (len(current_z_sliced)-1) * path_overlap + path_length != current_z.shape[1]:
                    # shifts[layer] = path_length * (len(current_z_sliced) + 1) - current_z.shape[1]
                    # shifts[layer] = (current_z.shape[1] - path_length) - (len(current_z_sliced) - 1 ) * path_length
                    shifts[layer] = (current_z.shape[1] - path_length) - (len(current_z_sliced) - 1 ) * path_overlap
                    current_z_sliced.append(current_z[:, -path_length:])
                n_seq = len(current_z_sliced)
                current_z_sliced = torch.cat(current_z_sliced, dim=0)
                current_z = current_z_sliced
            else:
                n_seq = 1
            #if clear:
            #    self.decoders[layer].clear()
            #pdb.set_trace()
            '''
            #pdb.set_trace()
            path_length = self.phidden[layer].get('path_length', current_z.shape[1])
            n_seq = math.ceil(current_z.shape[1]/path_length)
            if current_z.shape[1] != n_seq * path_length:
                device = next(self.parameters()).device
                current_z = torch.cat([current_z, torch.zeros(current_z.shape[0], n_seq * path_length - current_z.shape[1], *current_z.shape[2:]).to(device)], dim=1)
            current_z = current_z.contiguous().view(current_z.shape[0]*n_seq, path_length, *current_z.shape[2:])

            return_hidden = self.requires_recurrent and layer == len(self.platent) - 1
            current_out = self.encoders[layer](current_z, y=y, clear=clear, return_hidden = return_hidden)
            current_out['out_params'] = current_out['out_params'].view(n_batches, n_seq, self.platent[layer]['dim'])
            if current_out.get('out') is None:
                current_out['out'] = apply_method(current_out['out_params'], 'rsample')
            if return_hidden:
                current_out['hidden'] = current_out['hidden'].reshape(n_batches, n_seq, *current_out['hidden'].shape[1:])
            outs.append(current_out)
        #for i, out in enumerate(outs):
        #    print('encoding %d : '%i,out['out_params'].mean.std(0))
        if return_shifts:
            return outs, shifts
        else:
            return outs

    def decode(self, z, y=None, sample=True, from_layer=-1, shifts=None, n_steps=None, target_seq=None, clear=True, *args, **kwargs):
        # init full z input
        z = checklist(z); shifts = checklist(shifts, len(self.platent))
        z_all = [None] * len(self.platent);
        if from_layer < 0:
            from_layer = len(self.platent) + from_layer
        for i, z_tmp in enumerate(z):
            z_all[from_layer - len(z) + 1 + i] = z[i]

        current_z = z_all[from_layer]
        outs = []; n_batch = z_all[from_layer].shape[0]; n_seq = z_all[from_layer].shape[1]

        ''' 
        n_batch, n_seq = z[-1].shape[:2]
        current_z = z[-1].view(n_batch*n_seq, *z[-1].shape[2:])
        current_out = self.decoders[1](current_z, n_steps=1, clear=True, sample=True)
        current_out['out'] = current_out['out'].contiguous().view(n_batch, n_seq, *current_out['out'].shape[2:])
        current_out['out_params'] = current_out['out_params'].view(n_batch, n_seq, *current_out['out'].shape[2:])
        outs.append(current_out)
        current_z = current_out['out']

        #current_z = z[0]
        '''

        steps = [];
        if target_seq is None and z_all[0] is not None:
                target_seq = z_all[0].shape[1]
        if target_seq:
            cum_size = target_seq
            for i in range(1, len(z_all)):
                previous_step = self.phidden[i].get('path_length') or cum_size
                steps.append(previous_step)
                cum_size = int(ceil(cum_size / steps[-1]))
        else:
            steps = [self.phidden[i].get('path_length', 1) for i in range(1, len(self.phidden))]


        # init number of steps for each layer
        '''
        steps = []
        for i in range(1, len(self.platent)):
            if z_all[i-1] is not None:
                # steps.append(z_all[i-1].shape[1])
                steps.append(int(np.ceil(z_all[i-1].shape[1]/self.phidden[i].get('path_length', 1))))
            elif self.phidden[i].get('path_length'):
                steps.append(self.phidden[i].get('path_length'))
            else:
                if len(steps) > 0:
                    steps.append(steps[-1])
                else:
                    steps.append(target_seq)
        '''

        logger = GPULogger(verbose=False)
        logger('start decoding')
        for layer in reversed(range(1, from_layer+1)):
            # get number of steps to be decoded by RVAE decoder
            current_z = current_z.reshape(n_batch*n_seq, *current_z.shape[2:])
            n_overlap = self.phidden[layer].get('path_overlap') or n_steps
            # forward
            #if clear:
            #    self.decoders[layer].clear()
            current_out = self.decoders[layer](current_z, sample=True, n_steps=steps[layer-1], clear=clear)
            logger('layer %d forwarded'%layer)
            #TODO here we shall implement the different fusion modes
            if steps[layer-1] is not None:
                # if obtained from an encoded sequence, recover original shifts lost during encoding
                shift = shifts[layer] if shifts[layer] is not None else n_overlap
                # unflatten sequence
                if current_out.get('out') is None:
                    current_out['out'] = apply_method(current_out['out_params'], 'rsample')
                current = current_out['out'].reshape(n_batch, n_seq, steps[layer-1], self.platent[layer-1]['dim'])
                if n_seq >= 2:
                    # take two last sequences and merge them according to the shifts (default, concatentate)
                    current_1 = current[:, :-2, :n_overlap]; current_1 = current_1.contiguous().view(n_batch, current_1.shape[1]*current_1.shape[2], *current_1.shape[3:])
                    current_2 = current[:, -2:]; current_2 = torch.cat([current_2[:,0,:shift], current_2[:,1]], dim=1)
                    current_out['out'] = torch.cat([current_1, current_2], dim=1)
                # for distributions
                current = current_out['out_params'].reshape(n_batch, n_seq, steps[layer-1], self.platent[layer-1]['dim'])
                if n_seq >= 2:
                    current_1 = current[:, :-2, :n_overlap]; current_1 = current_1.view(n_batch, current_1.batch_shape[1]*current_1.batch_shape[2], *current_1.batch_shape[3:])
                    current_2 = current[:, -2:]; current_2 = cat_distribution([current_2[:,0,:shift], current_2[:,1]], dim=1)
                    current_out['out_params'] = cat_distribution([current_1, current_2], dim=1)
                else:
                    current_out['out_params'] = current.reshape(n_batch, steps[layer-1], self.platent[layer-1]['dim'])
                # x_1 = x_1[:, -2, :n_overlap].reshape(n_batch)
                # current_out['out_params'] = reshape_distribution(current_out['out_params'], (n_batch, n_seq, n_steps, self.platent[layer-1]['dim']))
                if z_all[layer-1] is not None:
                    if z_all[layer-1].shape[1] != current_out['out'].shape[1]:
                        current_out['out'] = current_out['out'][:, :z[layer-1].shape[1]]
                        current_out['out_params'] = current_out['out_params'][:, :z[layer-1].shape[1]]
            outs.append(current_out)

            logger('layer fusionned')
            # mix between encoder's z and predicted z
            if z_all[layer-1] is not None:
                epoch = kwargs.get('epoch')
                teacher_prob = self.teacher_prob if (self.teacher_warmup == 0 or epoch is None) else max(1 - (1 - self.teacher_prob)*epoch/self.teacher_warmup, self.teacher_prob)
                teacher_prob = torch.tensor(teacher_prob, device=current_out['out'].device).float()
                mask = dist.Bernoulli(teacher_prob).sample(sample_shape=[current_out['out'].shape[0]])
                current_z = crossed_select(mask, current_out['out'], z_all[layer-1])
            else:
                current_z = current_out['out']
            logger('layer crossed')
            n_seq =  current_z.shape[1]

        if self.decoders[0].take_sequences:
            current_out = self.decoders[0](current_z.contiguous().view(current_z.shape[0]*current_z.shape[1], *current_z[2:]))
        else:
            #TODO dont work with multihed output. check that
            original_shape = current_z.shape[:2]
            current_out = self.decoders[0](current_z.contiguous().view(current_z.shape[0]*current_z.shape[1], *current_z.shape[2:]))
            #current_out = self.decoders[0](current_z)
            current_out['out_params'] = current_out['out_params'].view(original_shape[0],original_shape[1],*current_out['out_params'].batch_shape[1:])
            if current_out.get('out') is None:
                current_out['out'] = apply_method(current_out['out_params'], 'rsample')
        logger('last layer decoded')

        outs.append(current_out)
        if target_seq is not None:
            if len(outs) > 1:
                outs[-2]['out'] = outs[-2]['out'][:, :target_seq]
                outs[-2]['out_params'] = outs[-2]['out_params'][:, :target_seq]
            outs[-1]['out'] = outs[-1]['out'][:, :target_seq]
            outs[-1]['out_params'] = outs[-1]['out_params'][:, :target_seq]

        return list(reversed(outs))


    def forward(self, x, y=None, options={}, *args, **kwargs):
        # formats input data
        logger = GPULogger(verbose=False)
        logger('init')
        x = self.format_input_data(x, pinput=self.pinput)
        logger('data formatted')
        # encode
        enc_out, true_lengths = self.encode(x, y=y, **kwargs)
        logger('data encoded')
        dec_out = self.decode([z['out'] for z in enc_out], y=y, shifts=true_lengths, **kwargs)
        logger('data decoded')

        x_params = dec_out[0]['out_params']
        dec_out = denest_dict(dec_out[1:]) if len(dec_out) > 1 else {}
        enc_out = denest_dict(enc_out)

        out = {'x_params':x_params, "logdets":enc_out.get('logdet'), "z_preflow":enc_out.get('out_preflow'),
                'z_params_dec':dec_out.get('out_params'), 'z_dec':dec_out.get('out'),
                'z_params_enc':enc_out['out_params'], 'z_enc':enc_out['out']}
        if self.requires_recurrent:
            out['recurrent_out'] = enc_out['hidden'][-1]

        return out


    def init_optimizer(self, optim_params, init_scheduler=True):
        optimization_mode = optim_params.get('optimize', 'full')
        if optimization_mode == 'recurrent':
            alg = optim_params.get('optimizer', 'Adam')
            optim_args = optim_params.get('optim_args', {'lr':1e-3})
            parameters = nn.ParameterList(sum([list(d.parameters()) for d in self.encoders[1:]] + [list(d.parameters()) for d in self.decoders[1:]], []))
            self.optimizers = {'default':getattr(torch.optim, alg)([{'params':parameters}], **optim_args)}
            if init_scheduler:
                self.init_scheduler(optim_params)
        else:
            super(ShrubVAE, self).init_optimizer(optim_params)


    def init_scheduler(self, optim_params):
        optimization_mode = optim_params.get('optimize', 'full')
        if optimization_mode == 'recurrent':
            scheduler = optim_params.get('scheduler', 'ReduceLROnPlateau')
            scheduler_args = optim_params.get('scheduler_args', {'patience':100, "factor":0.2, 'eps':1e-10})
            self.schedulers = {'default':getattr(torch.optim.lr_scheduler, scheduler)(self.optimizers['default'], **scheduler_args)}
        else:
            super(ShrubVAE, self).init_scheduler(optim_params)
