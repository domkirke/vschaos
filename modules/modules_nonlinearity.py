import torch, pdb
import torch.nn as nn
from . import init_module


class ScaledSoftsign(nn.Module):
    def __init__(self, device=None):
        super(ScaledSoftsign, self).__init__()
        self.params = nn.ParameterDict({'a': nn.Parameter(torch.tensor(2., requires_grad=True)),
                                        'b': nn.Parameter(torch.tensor(1., requires_grad=True))})
        nn.init.normal_(self.params['a'])
        nn.init.normal_(self.params['b'])

    def forward(self, x):
        return (self.params['a'] * x)/(1 + torch.abs( self.params['b'] * x) )


