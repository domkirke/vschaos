import pdb
from torch import zeros, ones, eye
from . import Bernoulli, Normal, MultivariateNormal, RandomWalk
from .distribution_flow import Flow


def IsotropicGaussian(batch_size, device="cpu", requires_grad=False):
    return Normal(zeros(*batch_size, device=device, requires_grad=requires_grad),
            ones(*batch_size, device=device, requires_grad=requires_grad))

def WienerProcess(batch_size, device="cpu", requires_grad=False):
    return RandomWalk(zeros(*batch_size, device=device, requires_grad=requires_grad),
            ones(*batch_size, device=device, requires_grad=requires_grad))

def IsotropicMultivariateGaussian(batch_size, device="cpu", requires_grad=False):
    return MultivariateNormal(zeros(*batch_size, device=device, requires_grad=requires_grad),
            covariance_matrix=eye(*batch_size, device=device, requires_grad=requires_grad))


def get_default_distribution(distrib_type, batch_shape, device="cpu", requires_grad=False):
    if issubclass(type(distrib_type), Flow):
        distrib_type = distrib_type.dist
    if distrib_type == Normal:
        return IsotropicGaussian(batch_shape, device=device, requires_grad=requires_grad)
    if distrib_type == MultivariateNormal:
        return IsotropicMultivariateGaussian(batch_shape, device=device, requires_grad=requires_grad)
    if distrib_type == RandomWalk:
        return WienerProcess(batch_shape, device=device, requires_grad=requires_grad)
    else:
        raise TypeError("Unknown default distribution for distribution %s"%distrib_type)
