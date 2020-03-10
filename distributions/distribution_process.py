import torch
from torch.distributions import kl
from . import Normal

def get_process_from_normal(normal_dist):
    mean = torch.cat([normal_dist.mean[:, 0], normal_dist.mean[:, 1:] - normal_dist.mean[:, -1]], axis=1)
    variance = torch.cat([normal_dist.variance[:, 0], normal_dist.variance[:, 1:] - normal_dist.variance[:, -1]], axis=1)
    return RandomWalk(mean, variance.sqrt())

# Trick described in Bayer & al.
class RandomWalk(Normal):
    def __init__(self, *args, validate_args=None, diagonal=True):
        if len(args)==1:
            process = get_process_from_normal(args[0])
            super(RandomWalk, self).__init__(process.mean, process.stddev, validate_args=validate_args)
        if len(args)==2:
            loc, scale = args
            assert len(loc.shape) > 2, "locations for Random Walk object must have ndims > 2"
            super(RandomWalk, self).__init__(loc, scale, validate_args=validate_args)
        self.diagonal = diagonal

    def sample(self, sample_shape=torch.Size()):
        diffs = Normal(self.mean, self.stddev).sample(sample_shape)
        return torch.cumsum(diffs, 1)

    def rsample(self, sample_shape=torch.Size()):
        diffs = Normal(self.mean, self.stddev).rsample(sample_shape)
        return torch.cumsum(diffs, 1)

    def log_prob(self, value):
        assert len(value.shape) > 2, "input value must have ndim > 2, got shape %s"%value.shape
        value_diffs = torch.cat([value[:, 0], value[:, 1:] - value[:, -1]], axis=1)
        return super().log_prob(value_diffs)

@kl.register_kl(RandomWalk, RandomWalk)
def kl_weiner_weiner(p, q):
    return kl._kl_normal_normal(p, q)

@kl.register_kl(RandomWalk, Normal)
def kl_weiner_normal(p, q):
    return kl._kl_normal_normal(p, get_process_from_normal(q))

@kl.register_kl(Normal, RandomWalk)
def kl_normal_weiner(p, q):
    return kl._kl_normal_normal(get_process_from_normal(p), q)
