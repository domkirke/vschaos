from . import Criterion
import torch, pdb
import torch.nn as nn
from ..modules.modules_bottleneck import MLP
from numpy import cumprod


class Adversarial(Criterion):
    module_class = MLP
    def __init__(self, latent_params=None, adversarial_params=None, optim_params=None, cuda=None, **kwargs):
        assert latent_params
        super(Adversarial, self).__init__()
        self.adversarial_params=adversarial_params
        self.latent_params = latent_params
        self.init_modules(latent_params, adversarial_params)
        self.init_optimizer(optim_params)
        if cuda is not None:
            if issubclass(type(cuda), list):
                self.cuda(cuda[0])
                if len(cuda) > 1:
                    self = vschaos.DataParallel(self, cuda, cuda[0])
            else:
                self.cuda(cuda)


    def init_modules(self, input_params, adversarial_params):
        adversarial_params = adversarial_params or {'dim':500, 'nlayers':2}
        self.hidden_module = MLP(input_params, adversarial_params)
        self.discriminator = nn.Linear(adversarial_params['dim'], 1)

    def init_optimizer(self, optim_params={}):
        self.optim_params = optim_params
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-3})

        optimizer = getattr(torch.optim, alg)([{'params':self.parameters()}], **optim_args)
        self.optimizer = optimizer

    def loss(self, params1, params2, sample=False, **kwargs):
        z_fake = params1.sample().float()
        z_real= params2.sample().float()
        if len(z_fake.shape) > 2:
            z_fake = z_fake.contiguous().view(cumprod(z_fake.shape[:-1])[-1], z_fake.shape[-1])
            z_real = z_real.contiguous().view(cumprod(z_real.shape[:-1])[-1], z_real.shape[-1])
        d_real = torch.sigmoid(self.discriminator(self.hidden_module(z_real)))
        d_fake = torch.sigmoid(self.discriminator(self.hidden_module(z_fake)))
        device = z_fake.device
        loss_real = torch.nn.functional.binary_cross_entropy(d_real, torch.ones(d_real.shape, device=device))
        loss_fake = torch.nn.functional.binary_cross_entropy(d_fake, torch.zeros(d_fake.shape, device=device))
        loss = -torch.mean(loss_real+loss_fake)
        return loss, (loss,)

    def get_named_losses(self, losses):
        return {'adversarial':losses[0]}

    def step(self, *args, **kwargs):
        self.optimizer.step()


