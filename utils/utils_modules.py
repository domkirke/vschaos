import torch
import torch.nn as nn
import pdb
from functools import reduce


class Identity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class Flatten(nn.Module):
    def forward(self, x, *args, **kwargs):
        dim = int(reduce(lambda x, y: x*y, x.shape[1:]))
        return x.view(x.shape[0], dim)

class Squeeze(nn.Module):
    def __repr__(self):
        return "Squeeze()"

    def forward(self, x, *args, **kwargs):
        return torch.squeeze(x)

class Unsqueeze(nn.Module):
    def __repr__(self):
        return "Unsqueeze(dim=%s)"%self.dim
    def __init__(self, dim=1):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x, *args, **kwargs):
        return torch.unsqueeze(x, self.dim)


class Reshape(nn.Module):
    def __repr__(self):
        return "Reshape(shape=%s)"%str(self.shape)

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = tuple([int(s) for s in shape])

    def forward(self, x, *args, **kwargs):
        shape = (*x.shape[0:-1], *self.shape)
        return torch.reshape(x, shape)
