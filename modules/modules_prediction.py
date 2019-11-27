import torch, torch.nn as nn, pdb
from . import flow
from .modules_bottleneck import MLP
from .modules_recurrent import RecurrentModule, RNNLayer, GRULayer, LSTMLayer, VRNNEncoder
from ..distributions import Normal, Empirical, FlowDistribution
from ..utils import checklist, print_stats, flatten_seq_method, print_module_stats, oneHot
from . import Sequential



# Baseline prediction
class RNNPrediction(nn.Module):
    RecurrentLayer = RNNLayer
    def __init__(self, latent_params, prediction_params):
        input_dim = latent_params[-1].get('dim')
        hidden_dim = prediction_params.get('dim', 200)
        #self.recurrent_module = prediction_params.get('class', self.RecurrentLayer)(input_dim, hidden_dim)
        self.recurrent_module = prediction_params.get('class', self.RecurrentLayer)(input_dim, hidden_dim)
        self.out_linear = MLP(hidden_dim, input_dim)
        self.n_predictions = prediction_params.get('n_predictions')
        self.n_steps = prediction_params.get('n_predictors')

    def forward(self, input, **kwargs):
        z_in = input['z_enc'][-1][:, -1]
        outs = []
        for i in range(self.n_steps):
            current_out = self.out_linear(self.recurrent_module(z_in, retain_hidden=True))
            outs.append(current_out)
        return torch.stack(outs, 1)

# Prediction modules with Contrastive Prediction Coding
# CPC modules. Here are different models for density ratio f(x, c)

class LinearCPC(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, False)
        nn.init.xavier_normal_(self.weight)

    def get_density_ratio(self, z, context, make_negatives=True):
        '''
        if make_negatives:
            bs = z.shape[0]
            pdb.set_trace()
            log_ratios = torch.cat([torch.nn.functional.bilinear(z[i].unsqueeze(0).repeat(bs, 1), context, self.weight.unsqueeze(0)) for i in range(bs)], dim=1)
            # re-order to put positive example at [:,0]
            device = log_ratios.device
            log_ratios = torch.cat([torch.masked_select(log_ratios, torch.eye(bs, device=device).byte()).unsqueeze(1),
                                    torch.masked_select(log_ratios, (1-torch.eye(bs, device=device)).byte()).view(bs,bs-1)], dim=1)
            print_stats(z, "z")
            print_stats(context, "c")
            print_stats(log_ratios, "ratios")
            print_stats(self.weight, 'w')
        else:

        '''
        log_ratios = torch.nn.functional.bilinear(z, context, self.weight.unsqueeze(0))
        return log_ratios


class CPCEncoder(nn.Module):
    def __init__(self, latent_params, prediction_params):
        super().__init__()
        n_layers = prediction_params.get('nlayers', 1)
        dims = [latent_params.get('dim')]+[prediction_params.get('hdim', 100)]*n_layers
        layer_class = prediction_params.get('layer', GRULayer)
        '''
        if prediction_params['layer'] == RNNLayer:
            prediction_params['nn_lin'] = "relu"
        '''
        layer_class = GRULayer
        self.recurrent_module = layer_class(latent_params['dim'], prediction_params.get('hdim', 100), num_layers=prediction_params.get('num_layers', 1), nn_lin=prediction_params.get('nn_lin'), batch_norm=prediction_params.get('batch','none'))
        self.linear_module = nn.Linear(dims[-1], prediction_params.get('dim', 32))
        #nn.init.normal_(self.linear_module.weight)
        nn.init.xavier_normal_(self.linear_module.weight)


    def clear(self):
        self.recurrent_module.clear()

    def flatten_parameters(self):
        self.recurrent_module.flatten_parameters()

    def forward(self, input, clear=True, **kwargs):
        if clear:
            self.clear()
        n_seq = input.shape[1]

        outputs = self.linear_module(self.recurrent_module(input))
        #for i in range(n_seq):
        #    outputs.append(self.linear_module(self.recurrent_module(input[:, i])).unsqueeze(1))
        return outputs
        #return torch.cat(outputs, dim=1)


def arange_exclusive(*args, excl=None):
    assert excl is not None, 'arange needs excl keyword'
    if len(args)==1:
        return torch.cat([torch.arange(0, excl), torch.arange(excl+1, args[0])])
    elif len(args)==2:
        return torch.cat([torch.arange(args[0], excl), torch.arange(excl+1, args[1])])
    else:
        raise TypeError('arange_exclusive expected 1 or 2 arguments, not more')


class CPCPredictiveLayer(nn.Module):
    RecurrentLayer = RNNLayer 
    ContextLayer = LinearCPC
    requires_recurrent = False
    encode_predictions = True

    def __init__(self, latent_params, prediction_params, **kwargs):
        super(CPCPredictiveLayer, self).__init__()

        # make autoregressive modules
        prediction_params['layer'] = prediction_params.get('layer') or self.RecurrentLayer
        recurrent_params = {'dim':prediction_params.get('hdim'), 'nlayers':prediction_params.get('ar_layers'), 'nn_lin':prediction_params.get('nn_lin')}
        if prediction_params.get('label_params'):
            self.label_params = prediction_params['label_params']
            self.recurrent_module = CPCEncoder({**latent_params, 'dim':latent_params['dim']+self.label_params['dim']}, prediction_params)
        else:
            self.recurrent_module = CPCEncoder(latent_params, prediction_params)
        self.recurrent_module.flatten_parameters()

        # make context embedding
        contextClass = prediction_params.get('cpc_class') or self.ContextLayer
        n_predictions = prediction_params.get('n_predictions')
        assert n_predictions, "CPCPredictiveLayer needs the prediction scope in advance"
        self.n_predictions = n_predictions
        self.n_predictors = prediction_params.get('n_predictors') or self.n_predictions
        predictors = [ ]
        for n in range(self.n_predictors):

            predictor = contextClass(prediction_params['dim'], latent_params['dim'])
            #predictor = nn.Sequential(predictor, nn.BatchNorm1d(num_features=latent_params['dim']))
            predictors.append(predictor)
        self.predictors = nn.ModuleList(predictors)

    '''
    def get_density_ratio(self, z, context, make_negatives=True):
        print_stats(context, 'context')
        print_stats(z, 'z')
        # get postive examples
        n_batches = z.shape[0]; n_seq = z.shape[1]
        positive_ratios = torch.cat([self.predictors[i].get_density_ratio(z[:, i], context[:, -1]).unsqueeze(1) for i in range(n_seq)], dim=1)
        # get negative examples
        negative_z = [ z[arange_exclusive(n_batches, excl=i)][torch.randperm(n_batches-1)][:,torch.randperm(n_seq)] for i in range(n_batches) ] # scramble sequences
        negative_ratios = []
        for b in range(n_batches):
            negative_ratios.append(torch.cat([self.predictors[i].get_density_ratio(negative_z[b][:, i], context[b, -1].unsqueeze(0).repeat(n_batches-1, 1)).unsqueeze(1) for i in range(n_seq)], dim=1).squeeze().t())

        return torch.cat([positive_ratios, torch.stack(negative_ratios)], dim=2)
    '''

    def get_density_ratio(self, z_predicted, z_real):
        density_ratios = []; n_seq = z_predicted.shape[1]
        for n in range(n_seq):
            density_ratios.append(torch.mm(z_real[:, n], z_predicted[:, n].t()))
            #nces.append(torch.diag(nn.LogSofmax(z_prod)))
        return torch.stack(density_ratios)

    def forward(self, out, clear=True, **kwargs):
        context_length = out['z_enc'][-1].shape[1] - self.n_predictors
        z_in = out['z_enc'][-1][:, :context_length]
        #print(out['z_enc'][-1].mean((0, 1)))

        if hasattr(self, 'label_params'):
            y = kwargs.get('y'),
            assert y, "needs metadata information if conditioning is enabled"
            y = list(y[0].values()); y_in = []
            for y_tmp in y:
                if len(y_tmp.squeeze().shape) == 1:
                    y_tmp = oneHot(y_tmp, self.label_params['dim'])
                device = next(self.parameters()).device
                if len(y_tmp.shape)==2:
                    y_tmp = y_tmp.unsqueeze(1).repeat(1,z_in.shape[1],1)
                y_tmp = y_tmp.to(device)
                y_in.append(y_tmp)
            z_in = torch.cat([z_in, *tuple(y_in)], dim=-1)
        contexts = self.recurrent_module(z_in)
        predictions = []
        for n in range(self.n_predictors):
            predictions.append(self.predictors[n](contexts[:,-1]))
        predictions = torch.stack(predictions, dim=1)
        #print_stats(z_in, 'z_in')
        #print_stats(predictions, 'preds')
        #print_stats(contexts, 'contexts')
        #pdb.set_trace()
        #print(predictions.mean((0, 1)))
        return {"out": predictions, "cpc_states":contexts}


# Flow-based prediction modules

class FlowPrediction(nn.Module):
    encode_predictions = True
    requires_recurrent = False
    def __init__(self, input_params, prediction_params, hidden_params=None, recurrent_params=None, **kwargs):
        super(FlowPrediction, self).__init__(**kwargs)
        # Define type of flow
        # blocks =
        blocks = prediction_params.get('blocks', [flow.PlanarFlow])
        for i in range(len(blocks)):
            if type(blocks[i]) == str:
                blocks[i] = getattr(flow, blocks[i])
        # get number of predictions
        self.n_predictions = prediction_params.get('n_predictions')
        assert self.n_predictions, "FlowPrediction needs the prediction scope in advance"
        # set amortization
        amortization = prediction_params.get('amortization', 'none')
        self.amortization = amortization
        amortize_dim = None
        if not amortization in ['none', None]:
            if amortization in ['input']:
                amortize_dim = input_params['dim']
                amortization = 'input'
            elif amortization in ['recurrent']:
                #assert recurrent_params
                amortize_dim = 128
                self.recurrent_embedding = nn.GRU(input_params['dim'], 128, batch_first=True)
                #self.recurrent_smoother = nn.BatchNorm1d(recurrent_params['dim'])
                #self.recurrent_smoother = nn.Sequential(nn.Linear(recurrent_params['dim'], recurrent_params['dim']), nn.BatchNorm1d(recurrent_params['dim']))
                self.requires_recurrent = True
                amortization="auxiliary"
            elif amortization in ['hidden']:
                assert hidden_params
                amortize_dim = hidden_params
                amortization='auxiliary'

        flows = []
        n_predictors = prediction_params.get('n_predictors') or self.n_predictions
        base_dist = Normal(torch.zeros(input_params['dim']), torch.eye(input_params['dim']))
        flows = flow.NormalizingFlow(dim=input_params['dim'], blocks=blocks, flow_length=n_predictors,
                                     density=base_dist, amortize_dim=amortize_dim, amortized=amortization)
        self.n_predictors = n_predictors
        self.flow = flows
        self.latent_size = input_params['dim']

    def forward(self, out, **kwargs):
        # data_in = out['z_enc'][-1]
        aux_in = None
        if self.amortization in ['recurrent']:
            aux_in, _ = self.recurrent_embedding(out['z_enc'][-1])
            aux_in = aux_in[:, -1]
            #print_stats(aux_in.grad)
            #aux_in = self.recurrent_smoother(out['recurrent_out'][:, -1])
        elif self.amortization in ['hidden']:
            aux_in = out['hidden'][-1]
        final_jacobs = []
        z_outs = []


        flow_dist = FlowDistribution(out['z_params_enc'][-1][:, -1], self.flow)
        # for n in range(len(self.flow)):
        #     z_out, log_jacobians = self.flow[n](data_in[:, context_in], aux=aux_in)
        #     final_jacobs.append(torch.sum(torch.cat(log_jacobians, dim=-1).unsqueeze(1), dim=-1))
        #     z_outs.append(z_out.unsqueeze(1))
        #     previous_z = z_outs[-1]
        #     if len(previous_z.shape) > 2:
        #         previous_z = previous_z.squeeze(-2)
        # preds = torch.cat(z_outs, dim=1)
        out, out_preflow = flow_dist.rsample(retain=True, aux_in=aux_in)
        return {'out_params':flow_dist, 'out':out, 'out_preflow':out_preflow}


## Gaussian processes predictions
L = 1e-2
def mse_kernel(x,y):
    return torch.exp(-0.5*torch.abs(x-y)**2/L)

def process_kernel(X,Y):
    out = torch.zeros((X.shape[0], Y.shape[0]), device=X.device);
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = mse_kernel(X[i], Y[j])
    return out

def sample_process(mean, cov):
    if mean.shape[1] > 1:
        return torch.stack([sample_process(mean[:,i].unsqueeze(1), cov) for i in range(mean.shape[1])], axis=-1)
    return torch.random.multivariate_normal(mean[:,0], cov)



class GPPrediction(nn.Module):
    encode_predictions = True
    requires_recurrent = False
    def __init__(self, input_params, prediction_params, **kwargs):
        super(GPPrediction, self).__init__()
        #self.register_parameter('sigma', nn.Parameter(torch.tensor(prediction_params.get('init_variance', 1e-5))))
        self.sigma = 1e-2
        self.timescale = 0.3
        #self.register_parameter('timescale', nn.Parameter(torch.tensor(prediction_params.get('timescale', 3e-1))))

        self.n_predictions = prediction_params.get('n_predictions')
        self.n_predictors = prediction_params.get('n_predictors', self.n_predictions)

    def forward(self, out, **kwargs):
        out = out['z_enc'][-1]
        t = torch.linspace(0, 1, out.shape[1]).to(out.device)
        context_in = out.shape[1] - self.n_predictors
        t_in  = t[:context_in]*self.timescale; t_pred = t[context_in:]*self.timescale
        out.mean(1); cov = process_kernel(t_in, t_in)

        k_pred = process_kernel(t_in, t_pred)
        k_mult = torch.mm(k_pred.t() , torch.inverse(cov + torch.eye(t_in.shape[0], device=out.device)*self.sigma))
        z_pred = torch.bmm(k_mult.unsqueeze(0).repeat(out.shape[0],1,1), out[:, :context_in])
        #print_stats(out, 'out')
        #print_stats(z_pred, 'pred')
        return {'out':z_pred} 




# configuring input data
# N = 60
# sigma = 1e-2
# sigma_pred = 1e-2
# t = np.linspace(0,1,N)
# sig = np.array([-0.3+2*np.sin(2*np.pi*0.5*t), 2-3*np.sin(2*np.pi*0.4*t + 1.4)]).T + sigma*np.random.randn(N,2)
#
# # get GP parmeters
# m = np.mean(sig, 0)
# cov = process_kernel(t, t)
#
# # perform prediction
# u = np.linspace(-4,4,20)
# k_test = process_kernel(t, u)
# k_mult = np.matmul(k_test.T , np.linalg.inv(cov + np.eye(N)*sigma_pred))
# new_mean = np.matmul(k_mult, sig)
# new_cov = process_kernel(u, u) - np.matmul(k_mult, k_test)
# outs= [sample_process(new_mean, new_cov) for i in range(3)]
#
# plt.plot(sig[:, 0], sig[:, 1])
# for out in outs:
#     plt.plot(out[:,0], out[:,1], linewidth=0.1)
# plt.plot(new_mean[:, 0],new_mean[:, 1])
# p
