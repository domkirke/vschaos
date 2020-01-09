import torch, pdb
from . import Criterion, reduce
from ..utils import checklist
from ..distributions import Distribution
import matplotlib.pylab as plt




class CriterionFunctional(Criterion):
    loss_fun = None

    def __init__(self, *args, **kwargs):
        super(CriterionFunctional, self).__init__(*args, **kwargs)

    def loss(self, params1=None, params2=None, input_params=None, *args, **kwargs):
        input_params = input_params or self.get_input_params(params1, params2)
        if issubclass(type(input_params), list):
            if not issubclass(type(params1), list):
                params1 = [params1]
            losses = tuple([self(params1[i], params2[i], input_params[i])[0] for i in range(len(input_params))])
            # losses = tuple([reduce(torch.nn.functional.binary_cross_entropy(x_params[i].mean, x, reduction='none'), self.reduction) for i in range(len(input_params))])
            loss = sum(losses)
        else:
            x = self.format_input(params1, params2)
            if issubclass(type(params1), Distribution):
                params1 = params1.rsample()
            if issubclass(type(params2), Distribution):
                params2 = params2.rsample()
            target = self.format_target(params2, input_params)
            loss = reduce(getattr(torch.nn.functional, self.loss_fun)(params1, params2, reduction='none'), self.reduction)
            losses = (float(loss),)
        return loss, losses

    def get_input_params(self, x_params, target):
        if issubclass(type(x_params), list):
            if issubclass(type(target), Distribution):
                target = [t.sample() for t in checklist(target)]
            assert issubclass(type(target), list)
            input_params = [{'dim':tuple(t.shape[1:])} for t in target]
        else:
            if issubclass(type(target), Distribution):
                target = target.sample()
            input_params = {'dim':tuple(target.shape[1:])}
        return input_params


    def format_input(self, x, input_params):
        if issubclass(type(x), torch.distributions.Distribution):
            if issubclass(type(x), torch.distributions.Categorical):
                return x.probs
            else:
                return x.mean
        else:
            return x

    def format_target(self, x, input_params):
        return x

    def get_named_losses(self, losses):
        names = ['%s_%d'%(self.loss_fun,i) if self._name is None else '%s_%s_%d'%(self.loss_fun, self._name, i) for i in range(len(losses))]
        return {names[i]:losses[i] for i in range(len(losses))}


class NLL(CriterionFunctional):
    loss_fun = 'nll_loss'

    def format_input(self, x, input_params):
        if issubclass(type(x), torch.distributions.Distribution):
            assert issubclass(type(x), torch.distributions.Categorical)
            return x.probs
        else:
            return x.mean

    def format_target(self, x, input_params):
        x = x.requires_grad_(False).long().squeeze()
        return x

class L1(CriterionFunctional):
    loss_fun = 'l1_loss'

class MSE(CriterionFunctional):
    loss_fun = 'mse_loss'




