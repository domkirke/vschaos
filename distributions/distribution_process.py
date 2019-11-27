import torch
from torch.distributions import kl
from . import Normal

def get_diff(tensor):
    if len(tensor.shape) == 2:
        dim = 0
    else:
        dim = 1
    tail = torch.index_select(tensor, dim, torch.range(0, tensor.shape[dim]-2))
    return tensor - torch.cat(tail, torch.index_select(torch.zeros_like(tail), dim, 0), dim=dim)


# Trick described in Bayer & al.
class RandomWalk(Normal):
    def sample(self, sample_shape=torch.Size()):
        diffs = Normal(self.mean, self.stddev).sample(sample_shape)
        return torch.cumsum(diffs, 1)

    def rsample(self, sample_shape=torch.Size()):
        diffs = Normal(self.mean, self.stddev).rsample(sample_shape)
        return torch.cumsum(diffs, 1)

def kl_weiner_normal(p, q):
    return kl._kl_normal_normal(p, q)

