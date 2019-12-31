from collections import OrderedDict
import torch, pdb
import torch.nn as nn
from . import flatten
from .modules_hidden import HiddenModule
from .modules_bottleneck import MLP
from . import Sequential
from .. import utils
from ..utils import concat_distrib, concat_tensors, print_stats, apply, apply_method, checklist


class RNNLayer(nn.Module):
    RecurrentCell = nn.RNN
    nn_lin = "relu"
    dump_patches = True
    def __init__(self, input_dim, output_dim, nn_lin="relu", batch_norm='batch', dropout=None, name_suffix="", *args,
                 **kwargs):
        super(RNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name_suffix = name_suffix
        self.batch_norm = batch_norm!="none"

        modules = OrderedDict()
        # init recurrent module
        nn_lin = nn_lin or self.nn_lin
        modules["recurrent" + name_suffix] = self.init_recurrent_module(input_dim, output_dim, nonlinearity=nn_lin, **kwargs)
        # Batch / Instance Normalization
        if batch_norm:
            if batch_norm == 'batch':
                modules["batch_norm_" + name_suffix] = nn.BatchNorm1d(output_dim)
            if batch_norm == 'instance':
                modules["instance_norm_" + name_suffix] = nn.InstanceNorm1d(1)

        # Dropout
        if not dropout is None:
            modules['dropout_' + name_suffix] = nn.Dropout(dropout)
        # get full module
        self.module = nn.Sequential(modules)
        self.previous_h = None
        self.flatten_parameters()

    def init_recurrent_module(self, input_dim, output_dim, **kwargs):
        module = self.RecurrentCell(input_dim, output_dim, batch_first=True, num_layers=kwargs.get('num_layers', 1), nonlinearity=kwargs.get('nonlinearity', 'relu'))
        for l in range(len(module._all_weights)):
            for weight in module._all_weights[l]:
                if 'weight_ih' in weight:
                    torch.nn.init.xavier_normal_(getattr(module, weight))
                elif 'weight_hh' in weight:
                    torch.nn.init.orthogonal_(getattr(module, weight))
                elif 'bias' in weight:
                    torch.nn.init.zeros_(getattr(module, weight))
        return module


    def to(self, *args, **kwargs):
        super(RNNLayer, self).cuda(*args, **kwargs)

    def flatten_parameters(self):
        self.module[0].flatten_parameters()

    def forward(self, x, previous_h=None, retain_hidden=False):
        if len(x.shape) > 3:
            x = flatten(x, 2)
        previous_h = previous_h or self.previous_h or None
        out, _ = self.module[0](x, previous_h)
        if self.batch_norm:
            if self.batch_norm == 'batch':
                out = self.module._modules['batch_norm_' + self.name_suffix](out)
            elif self.batch_norm == 'instnce':
                out = self.module._modules['instance_norm_' + self.name_suffix](out.unsqueeze(1))
                out = out.squeeze()
            # out = self.module._modules['nnlin' + self.name_suffix](out)
        if retain_hidden:
            self.previous_h = out
        return out

    def clear(self):
        self.previous_h = None

class GRULayer(RNNLayer):
    RecurrentCell = nn.GRU

    def init_recurrent_module(self, input_dim, output_dim, **kwargs):
        module = self.RecurrentCell(input_dim, output_dim, batch_first=True, num_layers=kwargs.get('num_layers', 1))
        for l in range(len(module._all_weights)):
            for weight in module._all_weights[l]:
                if 'weight_ih' in weight:
                    torch.nn.init.xavier_normal_(getattr(module, weight))
                elif 'weight_hh' in weight:
                    torch.nn.init.orthogonal_(getattr(module, weight))
                elif 'bias' in weight:
                    torch.nn.init.zeros_(getattr(module, weight))
        return module
    #     module = self.RecurrentCell(input_dim, output_dim)
    #     torch.nn.init.xavier_normal_(module.weight_ih)
    #     #torch.nn.init.orthogonal_(modules["recurrent" + name_suffix].weight_hh)
    #     torch.nn.init.xavier_normal_(module.weight_hh)
    #     torch.nn.init.zeros_(module.bias_hh)
    #     torch.nn.init.zeros_(module.bias_ih)
    #     return module


class LSTMLayer(RNNLayer):
    RecurrentCell = nn.LSTM

    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super(LSTMLayer, self).__init__(input_dim, output_dim, *args, **kwargs)
        self.return_cell = kwargs.get('return_cell', False)
        self.previous_c = None
        self.flatten_parameters()

    def forward(self, x, previous_h = None, previous_c = None, retain_hidden=False):
        previous_h = previous_h or self.previous_h or None
        previous_c = previous_c or self.previous_c or None
        if previous_h is None or previous_c is None:
            contexts = None
        else:
            contexts = (previous_h, previous_c)
        out_h, (h_n, c_n) = self.module._modules['recurrent' + self.name_suffix](x, contexts)
        if self.batch_norm:
            if self.batch_norm == 'batch':
                out_h = self.module._modules['batch_norm_' + self.name_suffix](out_h)
            elif self.batch_norm == 'instance':
                out_h = self.module._modules['instance_norm_' + self.name_suffix](out_h.unsqueeze(1))
                out_h = out_h.squeeze();
        '''
        if self.nn_lin:
            out = self.module._modules['nnlin' + self.name_suffix](out)
        '''
        if retain_hidden:
            self.previous_h = h_n
            self.previous_c = c_n
        if self.return_cell:
            return out_h, (h_n, c_n)
        else:
            return out_h

    def init_recurrent_module(self, input_dim, output_dim, **kwargs):
        module = self.RecurrentCell(input_dim, output_dim, batch_first=True)
        for l in range(len(module._all_weights)):
            for weight in module._all_weights[l]:
                if 'weight_ih' in weight:
                    torch.nn.init.xavier_normal_(getattr(module, weight))
                elif 'weight_hh' in weight:
                    torch.nn.init.orthogonal_(getattr(module, weight))
                elif 'bias' in weight:
                    torch.nn.init.zeros_(getattr(module, weight))
        return module


    def clear(self):
        super(LSTMLayer, self).clear()
        self.previous_c = None


class RecurrentModule(MLP):
    DefaultLayer = RNNLayer
    dump_patches = True

    def __init__(self, *args, **kwargs):
        super(RecurrentModule, self).__init__(*args, **kwargs)

    def clear(self):
        for i in range(self.phidden['nlayers']):
            getattr(self.hidden_module, 'layer_%d'%i).clear()

    def forward(self, x, y=None, sample=True, return_states=False, bypass_embedding=False, *args, **kwargs):
        if issubclass(type(x), list):
            is_seq = len(x[0].shape) >= 3
        else:
            is_seq = len(x.shape) >= 3
        if not is_seq:
            return super(RecurrentModule, self).forward(x, y=y, sample=sample, *args, **kwargs)
        else:
            # out = [None]*x.shape[1]
            # for i in range(x.shape[1]):
            #     if issubclass(type(x), list):
            #         x_in = [inp[:, i] for inp in x]
            #     else:
            #         x_in = x[:, i]
            #     out[i] = super(RecurrentModule, self).forward(x_in, y=y, sample=sample, *args, **kwargs)
            # if return_states:
            #     out = torch.stack(out, 1)
            # else:
            #     out = out[-1]
            out = super(RecurrentModule, self).forward(x, y=y, sample=sample, *args, **kwargs)
            return out

    def previous_h(self, batch_shape=None):
        previous_h = [None]*self.phidden['nlayers']
        for i in range(len(previous_h)):
            previous_h[i] = self.hidden_module._modules["layer_%d"%i].previous_h
            if previous_h[i] is None:
                previous_h[i] = torch.zeros(batch_shape or 1, self.hidden_module._modules["layer_%d"%i].output_dim, device=next(self.parameters()).device)
        return previous_h

    def flatten_parameters(self):
        for i in self.hidden_module:
            i.flatten_parameters()


class RVAEEncoder(HiddenModule):
    default_module = MLP
    default_recurrent = RecurrentModule
    take_sequences = True
    dump_patches = True
    def __init__(self, pins, phidden={}, pouts=None, recurrent_params={}, make_flows=None, *args, **kwargs):
        has_hidden = phidden is not None and phidden != {}

        self.precurrent = recurrent_params
        super(RVAEEncoder, self).__init__(pins, phidden, pouts=None, precurrent=recurrent_params, *args, **kwargs)

        self.pouts = pouts
        if pouts:
            self.out_modules = self.make_output_layers(recurrent_params, pouts)

        # if has_hidden:
        #     self.recurrent_modules = self.make_recurrent_layer(phidden, recurrent_params, *args, **kwargs)
        # else:
        # self.recurrent_modules = self.make_recurrent_layer(pins, recurrent_params, *args, **kwargs)

    def make_hidden_layers(self, pins, phidden={"dim": 800, "nlayers": 2, 'label': None, 'conditioning': 'concat'},
                           *args, **kwargs):
        phidden['batch_norm'] = False
        hidden_module =  super().make_hidden_layers(pins, phidden, *args, **kwargs)
        precurrent = kwargs.get('precurrent') or self.precurrent
        recurrent_module = self.make_recurrent_layer(phidden, precurrent)
        return Sequential(hidden_module, recurrent_module)

    def make_recurrent_layer(self, phidden, precurrent, *args, **kwargs):
        if issubclass(type(precurrent), list):
            assert len(precurrent) == len(phidden), "in case of individual recurrent layers, recurrent specifications must be equal to incoming hidden embeddings"
            return [self.make_recurrent_layer(phidden[i], precurrent[i]) for i in range(len(phidden))]
        recurrent_class = precurrent.get('class', self.default_recurrent)
        recurrent_module = recurrent_class(phidden, precurrent)
        recurrent_module.flatten_parameters()
        return recurrent_module

    @property
    def hidden_out_params(self, hidden_modules=None):
        hidden_modules = hidden_modules or self._hidden_modules
        if issubclass(type(hidden_modules), nn.ModuleList):
            params = []
            for i, m in enumerate(hidden_modules):
                if hasattr(hidden_modules[i], 'phidden'):
                    params.append(hidden_modules[i][1].phidden)
                else:
                    params.append(checklist(self.precurrent, n=len(hidden_modules))[i])
            return params
        else:
            if hasattr(hidden_modules, 'phidden'):
                return checklist(hidden_modules[1].phidden)[-1]
            else:
                return checklist(checklist(self.precurrent)[0])[-1]

    def clear_recurrent(self):
        self.hidden_modules[1].clear()

    def forward_hidden(self, x, y=None, clear=True, *args, **kwargs):
        if clear:
            self.clear_recurrent()
        recurrent_in = x
        # if hasattr(self, "hidden_modules"):
        module_out = super(RVAEEncoder, self).forward_hidden(recurrent_in, y=y, *args, **kwargs)
        # module_out = self.recurrent_modules(recurrent_in)
            # recurrent_in = x[:, j]
            # if hasattr(self, "hidden_modules"):
            #     recurrent_in = super(RVAEEncoder, self).forward_hidden(recurrent_in, y=y, *args, **kwargs)
            # if issubclass(type(self.precurrent), list):
            #     module_out = [self.recurrent_modules[i](recurrent_in[i]) for i in range(len(self.precurrent))]
            # else:
            #     module_out = self.recurrent_modules(recurrent_in)
        return module_out[:, -1]


class RVAEDecoder(HiddenModule):
    default_module = MLP
    default_recurrent = RecurrentModule
    take_sequences = True
    def __init__(self, pins, phidden={}, pouts=None, recurrent_params={}, *args, **kwargs):
        has_hidden = phidden is not None and phidden != {}
        self.precurrent = recurrent_params
        self.pouts = pouts
        super(RVAEDecoder, self).__init__(pins, phidden, pouts=pouts, precurrent=recurrent_params, *args, **kwargs)

        if pouts:
            if has_hidden:
                self.out_modules = self.make_output_layers(phidden, pouts, is_seq=True)
            else:
                self.out_modules = self.make_output_layers(recurrent_params, pouts, is_seq=True)

    def make_hidden_layers(self, pins, phidden={"dim": 800, "nlayers": 2, 'label': None, 'conditioning': 'concat'},
                           *args, **kwargs):
        precurrent = kwargs.get('precurrent') or self.precurrent
        hidden_module =  super().make_hidden_layers(precurrent, phidden, *args, **kwargs)
        recurrent_module = self.make_recurrent_layer(pins, precurrent)
        return Sequential(recurrent_module, hidden_module)

    @property
    def hidden_out_params(self, hidden_modules=None):
        hidden_modules = hidden_modules or self._hidden_modules
        if issubclass(type(hidden_modules), nn.ModuleList):
            params = []
            for i, m in enumerate(hidden_modules):
                if hasattr(hidden_modules[i], 'phidden'):
                    params.append(hidden_modules[i][1].phidden)
                else:
                    params.append(checklist(self.phidden, n=len(hidden_modules))[i])
            return params
        else:
            if hasattr(hidden_modules, 'phidden'):
                return checklist(hidden_modules[1].phidden)[-1]
            else:
                return checklist(checklist(self.phidden)[0])[-1]

    def make_recurrent_layer(self, phidden, precurrent, *args, **kwargs):
        if issubclass(type(precurrent), list):
            assert len(precurrent) == len(phidden), "in case of individual recurrent layers, recurrent specifications must be equal to incoming hidden embeddings"
            return [self.make_recurrent_layer(phidden[i], precurrent[i]) for i in range(len(phidden))]
        recurrent_module = precurrent.get('class', self.default_recurrent)
        return recurrent_module(phidden, precurrent)

    def forward_hidden(self, x, y=None, clear=True, *args, **kwargs):
        if clear:
            utils.apply_method(self.hidden_modules, "clear")
        if issubclass(type(self.precurrent), list):
            hidden_out = [self.hidden_modules[i](x[i]) for i in range(len(self.hidden_modules))]
        else:
            hidden_out = self.hidden_modules(x.contiguous())
        return hidden_out

    def clear_recurrent(self):
        self.hidden_modules[0].clear()

    def forward(self, x,  n_steps=100, y=None, sample=True, clear=True, return_hidden=False, *args, **kwargs):
        if clear:
            self.clear_recurrent()
        outs = {}

        hidden_outs = self.forward_hidden(x.unsqueeze(1).repeat(1,n_steps,1), y=y, clear=False)
        if return_hidden:
            outs['hidden'] = hidden_outs

        n_batch = hidden_outs.shape[0]; n_seq = hidden_outs.shape[1]
        outs['out_params'] = self.forward_params(hidden_outs.contiguous())
        outs['out_params'] = outs['out_params'].reshape(n_batch, n_seq, *outs['out_params'].batch_shape[2:])

        return outs






class VRNNEncoder(HiddenModule):

    def __init__(self, pins, phidden=None, pouts=None, recurrent_module=None, linked=True, *args, **kwargs):
        assert recurrent_module
        super().__init__(pins, phidden, pouts=None,  linked=linked, *args, **kwargs)
        self.out_modules = self.make_output_layers([phidden, recurrent_module.phidden], pouts)
        self.__setattr__('recurrent_module', recurrent_module, register_parameters=False)

    def __setattr__(self, name, value, register_parameters=True):
        if register_parameters:
            torch.nn.Module.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def forward_hidden(self, x, y=None, *args, **kwargs):
        hidden_out =  super().forward_hidden(x, y, *args, **kwargs)
        print(hidden_out)
        return hidden_out

    def forward_params(self, x, y=None, previous_h=None, *args, **kwargs):
        assert previous_h is not None
        return super().forward_params(torch.cat([x, previous_h], dim=-1), y=y)

    def forward(self, x, y=None, sample=True, update=True, previous_h=None, *args, **kwargs):
        outs = super().forward(x, y, sample, return_hidden=True, *args, **kwargs)
        outs['out'] = apply_method(outs['out_params'], 'rsample')

        outs = utils.merge_dicts(outs, dim=1, unsqueeze=1)
        recurrent_out = self.recurrent_module([outs[i]['hidden'], outs[i]['out']], bypass_embedding=True)
        recurrent_outs = concat_tensors(recurrent_out)
        return {**outs, 'recurrent':recurrent_outs}



class VRNNDecoder(HiddenModule):

    def __init__(self, pins, phidden=None, pouts=None, recurrent_module=None, prior_module=None, pflows=None, linked=True, *args, **kwargs):
        assert recurrent_module
        super().__init__([pins, recurrent_module.phidden], phidden, pouts=pouts, pflows=pflows, linked=linked, *args, **kwargs)
        self.__setattr__('recurrent_module', recurrent_module, register_parameters=False)
        self.__setattr__('prior_module', prior_module, register_parameters=False)

    def __setattr__(self, name, value, register_parameters=True):
        if register_parameters:
            torch.nn.Module.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def forward_hidden(self, x, y=None, previous_h=None, *args, **kwargs):
        assert previous_h is not None
        return super().forward_hidden(torch.cat([x, previous_h], dim=-1), y=y)

    def forward(self, x, y=None, sample=True, update=True, previous_h=None, n_seq=None, *args, **kwargs):
        n_seq = n_seq or x.shape[1]; outs = [None]*n_seq
        recurrent_out = [None]*n_seq; z_pred = []
        for i in range(n_seq):
            # get current hidden state
            if previous_h is None or i >= previous_h.shape[1]:
                h = self.recurrent_module.previous_h(batch_shape=x.shape[0])[-1]
            else:
                h = previous_h[:, i]
            # get hidden state
            if i >= x.shape[1]:
                z_pred.append(self.prior_module(h))
                z = z_pred[-1]['out']
            else:
                z = x[:,i]
            # forward!
            outs[i] = HiddenModule.forward(self, z, y=y, sample=sample, previous_h=h, *args, **kwargs)
            # record
            if previous_h is None or i >= previous_h.shape[1]:
                recurrent_out[i] = self.recurrent_module((outs[i]['out'], z))
            else:
                recurrent_out[i] = previous_h[:, i]

        if previous_h is not None:
            recurrent_outs = previous_h
        else:
            recurrent_outs = concat_tensors(recurrent_out)
        outs = utils.merge_dicts(outs, dim=1, unsqueeze=1)
        z_pred = utils.merge_dicts(z_pred, dim=1, unsqueeze=1)
        return {**outs, 'recurrent':recurrent_outs, 'z_pred':z_pred}




'''
class VRNNEncoder(HiddenModule):
    default_module = MLP
    dump_patches = True
    take_sequences = True

    def make_hidden_layers(self, pins, phidden={}, recurrent_params=None, *args, **kwargs):
        assert recurrent_params
        x_embedding_params = {**phidden, 'dim':phidden.get('embed_dims', 800), "nlayers":phidden.get('embed_nlayers', 1)}
        x_embedding = super(VRNNEncoder, self).make_hidden_layers(pins, x_embedding_params, *args, **kwargs)
        hidden_module = super(VRNNEncoder, self).make_hidden_layers([x_embedding_params, recurrent_params], {**phidden, 'class':self.default_module})
        return nn.ModuleDict({'embedding':x_embedding, 'hidden_module':hidden_module})

    @property
    def hidden_modules(self):
        return self._hidden_modules['embedding']

    @property
    def embedding_params(self):
        return self._hidden_modules['embedding'].phidden

    def forward_hidden(self, x, y=None, h=None, *args, **kwargs):
        out_embed = self._hidden_modules['embedding'](x)
        if h is None:
            h = torch.zeros(x.shape[0], self._hidden_modules['hidden_module'].pinput[1]['dim'], device=x.device)
        out_hidden = self._hidden_modules['hidden_module']([out_embed, h])
        return {'embed_out':out_embed, 'hidden_out':out_hidden}

    def forward_params(self, hidden, y=None, *args, **kwargs):
        return super(VRNNEncoder, self).forward_params(hidden['hidden_out'], y=y, *args, **kwargs)


class VRNNDecoder(VRNNEncoder):
    default_module = MLP
    dump_patches = True
    take_sequences = True

    def make_hidden_layers(self, pins, phidden={}, recurrent_params=None, *args, **kwargs):
        assert recurrent_params
        x_embedding_params = {**phidden, 'dim':phidden.get('embed_dims', 800), "nlayers":phidden.get('embed_nlayers', 1), 'class':self.default_module}
        x_embedding = HiddenModule.make_hidden_layers(self, pins, x_embedding_params)
        hidden_module = HiddenModule.make_hidden_layers(self, [x_embedding_params, recurrent_params], phidden, *args, **kwargs)
        return nn.ModuleDict({'embedding':x_embedding, 'hidden_module':hidden_module})

    @property
    def hidden_modules(self):
        return self._hidden_modules['hidden_module']
'''

#TODO the main difference in the Bayer article is FastDropout, implement it?

# class STORNEncoder(HiddenModule):
#     default_module = RecurrentModule
#     dump_patches = True
#     def make_hidden_layers(self,pins, phidden={"dim":800, "x_embed":800, "z_embed":800, "nlayers":2, 'label':None, 'conditioning':'concat'}, pouts=None, *args, **kwargs):
#         x_embed = phidden.get('x_embed', 800)
#         # make x embedding
#         xembed_params = dict(phidden); phidden['dim'] = x_embed; xembed_params['nlayers'] = 1
#         self.x_embedding = MLP(pins, xembed_params)
#
#         # init recurrent model
#         recurrent_pins = dict(phidden); recurrent_pins['dim'] = x_embed
#         return super(STORNEncoder, self).make_hidden_layers(recurrent_pins, phidden, *args, **kwargs)
#
#     def forward(self, x, y=None, sample=True, clear=True, *args, **kwargs):
#         if clear:
#             utils.apply_method(self.hidden_modules, "clear")
#         outs = {'hidden': [], 'out_params': [], 'out': []}
#         n_steps = x.shape[1]
#         current_device = next(self.parameters()).device
#         previous_h = torch.zeros((x.shape[0], self.phidden['dim']), device=current_device)
#         for j in range(n_steps):
#             x_embedded = self.x_embedding(x[:, j])
#             previous_h = self.forward_hidden(x_embedded, previous_h=previous_h)
#             outs['hidden'].append(previous_h)
#             z_params_out = self.forward_params(previous_h); outs['out_params'].append(z_params_out)
#             z_out = self.sample(z_params_out); outs['out'].append(z_out)
#         outs['out_params'] = concat_distrib(outs['out_params'])
#         outs['out'] = concat_tensors(outs['out'])
#         outs['hidden'] = concat_tensors(outs['hidden'])
#         return outs
#
#
# class STORNDecoder(VRNNDecoder):
#     default_module = RecurrentModule
#     dump_patches = True
#     def make_hidden_layers(self,pins, phidden={"dim":800, "nlayers":2, 'label':None, 'conditioning':'concat'}, pouts= None, encoder=None, *args, **kwargs):
#         pouts = pouts if not pouts is None else self.pouts
#         assert self.pouts
#         # some functions are shared with the VRNN encoder
#         self.x_embedding = encoder.x_embedding
#         # make z embedding
#         z_embed = phidden.get('z_embed', 800)
#         zembed_params = dict(phidden); phidden['dim'] = z_embed; zembed_params['nlayers'] = 1
#         self.z_embedding = MLP(pins, zembed_params)
#         recurrent_pins = dict(phidden); recurrent_pins['dim'] = encoder.x_embedding.phidden['dim'] + z_embed
#         return HiddenModule.make_hidden_layers(self, recurrent_pins, phidden, *args, **kwargs)

