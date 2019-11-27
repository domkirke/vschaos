import torch
from . import TransformedDistribution
from torch.distributions.utils import _sum_rightmost
# from ..modules.flow import flow


class FlowDistribution(TransformedDistribution):
    requires_preflow = True
    def __init__(self, base_distribution, flow, validate_args=None):
        super(FlowDistribution, self).__init__(base_distribution, flow.transforms, validate_args=validate_args)
        self.flow = flow

    def __repr__(self):
        return "FlowDistribution(%s, %s)"%(self.base_dist, self.flow)

    # def __getattr__(self, item):
    #     if hasattr(self, item):
    #         return super(FlowDistribution, self).__getattr__(item)
    #     else:
    #         return self.base_dist.__getattr__(item)

    def sample(self, sample_shape=torch.Size(), aux_in = None, retain=False):
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            x_0 = x
            if retain:
                full_x = []
            self.flow.amortization(x_0, aux=aux_in)
            for i, flow in enumerate(self.flow.blocks):
                x = flow(x)
                if retain:
                    full_x.append(x.unsqueeze(1))
            if retain:
                return torch.cat(full_x, dim=1), x_0
            else:
                return x, x_0

    def rsample(self, sample_shape=torch.Size(), aux_in=None, retain=False):
        x = self.base_dist.rsample(sample_shape)
        x_0 = x
        if retain:
            full_x = []
        self.flow.amortization(x_0, aux=aux_in)
        for i, flow in enumerate(self.flow.blocks):
            x = flow(x)
            if retain:
                full_x.append(x.unsqueeze(1))
        if retain:
            return torch.cat(full_x, dim=1), x_0
        else:
            return x, x_0


    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        log_prob = self.base_dist.log_prob(value) - self.flow.bijectors.log_abs_det_jacobian(value)
        return log_prob


class Flow(object):

    def __init__(self, dist_type, flow_module):
        super(Flow, self).__init__()
        self._dist = dist_type
        self._flow = flow_module

    @property
    def dist(self):
        return self._dist

    @property
    def has_rsample(self):
        return self._dist.has_rsample

    def __call__(self, *args, **kwargs):
        return FlowDistribution(self._dist(*args, **kwargs), self._flow)




