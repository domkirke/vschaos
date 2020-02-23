from . import Criterion
import torch, pdb
import torch.nn as nn
from .. import DataParallel
from .. import distributions as dist
from ..utils import checktuple, checklist
from ..modules.modules_bottleneck import MLP
from numpy import cumprod


class Adversarial(Criterion):
    module_class = MLP
    def __init__(self, input_params=None, adversarial_params=None, optim_params={}, cuda=None, **kwargs):
        assert input_params
        super(Adversarial, self).__init__()
        self.adversarial_params=adversarial_params
        self.input_params = input_params
        self.init_modules(input_params, adversarial_params)
        self.init_optimizer(optim_params)
        if cuda is not None:
            if issubclass(type(cuda), list):
                self.cuda(cuda[0])
                if len(cuda) > 1:
                    self = DataParallel(self, cuda, cuda[0])
            else:
                self.cuda(cuda)


    def init_modules(self, input_params, adversarial_params):
        adversarial_params = adversarial_params or {'dim':500, 'nlayers':2}
        self.hidden_module = MLP(input_params, adversarial_params)
        self.discriminator = nn.Linear(checklist(adversarial_params['dim'])[-1], 1)

    def init_optimizer(self, optim_params={}):
        self.optim_params = optim_params
        alg = optim_params.get('optimizer', 'Adam')
        optim_args = optim_params.get('optim_args', {'lr':1e-3})

        optimizer = getattr(torch.optim, alg)([{'params':self.parameters()}], **optim_args)
        self.optimizer = optimizer

    def loss(self, params1=None, params2=None, sample=False, **kwargs):
        assert params1 is not None
        assert params2 is not None

        #pdb.set_trace()
        if issubclass(type(params1), dist.Distribution):
            z_fake = params1.rsample().float()
        else:
            z_fake = params1.float()
        if issubclass(type(params2), dist.Distribution):
            z_real = params2.rsample().float()
        else:
            z_real = params2.float()

        data_dim = len(checktuple(self.input_params['dim'])) 
        if len(z_fake.shape) > data_dim + 1:
            z_fake = z_fake.contiguous().view(cumprod(z_fake.shape[:-data_dim])[-1], *z_fake.shape[-data_dim:])
            z_real = z_real.contiguous().view(cumprod(z_real.shape[:-data_dim])[-1], *z_real.shape[-data_dim:])

        device = z_fake.device
        # get generated loss
        d_gen = torch.sigmoid(self.discriminator(self.hidden_module(z_fake)))
        #print('d_gen : ', d_gen.min(), d_gen.max())
        try:
            assert d_gen.min() >= 0 and d_gen.max() <= 1
        except AssertionError:
            pdb.set_trace()
        loss_gen = self.reduce(torch.nn.functional.binary_cross_entropy(d_gen, torch.ones(d_gen.shape, device=device), reduction="none"))

        # get discriminative loss
        d_real = torch.sigmoid(self.discriminator(self.hidden_module(z_real)))
        #print('d_real : ', d_real.min(), d_real.max())
        try:
            assert d_real.min() >= 0 and d_real.max() <= 1
        except AssertionError:
            pdb.set_trace()

        d_fake = torch.sigmoid(self.discriminator(self.hidden_module(z_fake.detach())))
        #print('d_fake : ', d_fake.min(), d_fake.max())
        try:
            assert d_fake.min() >= 0 and d_fake.max() <= 1
        except AssertionError:
            pdb.set_trace()

        loss_real = torch.nn.functional.binary_cross_entropy(d_real, torch.ones(d_real.shape, device=device), reduction="none")
        loss_fake = torch.nn.functional.binary_cross_entropy(d_fake, torch.zeros(d_fake.shape, device=device), reduction="none")

        self.adv_loss = self.reduce((loss_real+loss_fake)/2)
        return loss_gen, (loss_gen.cpu().detach().numpy(), self.adv_loss.cpu().detach().numpy())

    def get_named_losses(self, losses):
        if issubclass(type(losses[0]), (tuple, list)):
            outs = {}
            for i,l in enumerate(losses):
                outs = {**outs, 'gen_loss_%d'%i:l[0], 'adv_loss_%d'%i:l[1]}
            return outs
        else:
            return {'gen_loss':losses[0], 'adv_loss':losses[1]}

    def step(self, *args, **kwargs):
        self.adv_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


