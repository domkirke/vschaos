import torch, torch.nn as nn
from .. import distributions as dist
import sklearn.manifold as manifold
import sklearn.decomposition as decomposition

from ..utils.misc import checklist

class DimRedBaseline(nn.Module):

    def __init__(self, input_params, latent_params, **kwargs):
        super(DimRedBaseline, self).__init__()
        self.pinput = input_params
        latent_params = checklist(latent_params)
        self.platent = checklist(latent_params)
        self.dimred_module = self.dimred_class(n_components = self.platent[-1]['dim'], **kwargs)

    def encode(self, x, **kwargs):
        input_device = torch.device('cpu')
        if torch.is_tensor(x):
            input_device = x.device
        if len(x.shape) > 2:
            x = x[:, 0]
        dimred_out = torch.from_numpy(self.dimred_module.fit_transform(x)).to(input_device).float()
        dimred_dist = dist.Normal(dimred_out, torch.zeros_like(dimred_out)+1e-12)
        return [{'out':dimred_out, 'out_params':dimred_dist}]

    def decode(self, z,squeezed=False, **kwargs):
        input_device = torch.device('cpu')
        if torch.is_tensor(z):
            input_device = z.device
            z = z.cpu().detach().numpy()
        dimred_out = torch.from_numpy(self.dimred_module.inverse_transform(z)).to(input_device).float()
        if squeezed:
            dimred_out = dimred_out.unsqueeze(1)
        dimred_dist = dist.Normal(dimred_out, torch.zeros_like(dimred_out)+1e-12)
        return [{'out':dimred_out, 'out_params':dimred_dist}]

    def forward(self, x, y=None, **kwargs):
        squeezed = False
        if len(x.shape) > 2:
            x = x[:, 0]
            squeezed = True
        z = self.encode(x, **kwargs)
        reconstruction= self.decode(z[0]['out'], squeezed = squeezed, **kwargs)
        return {'x_params': reconstruction[0]['out_params'],
                'z_params_enc':[z[0]['out_params']],
                'z_enc':[z[0]['out']],
                'z_params_dec':[], 'z_dec':[]}


class PCABaseline(DimRedBaseline):
    dimred_class = decomposition.PCA

class ICABaseline(DimRedBaseline):
    dimred_class = decomposition.FastICA
