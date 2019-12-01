import torch, matplotlib.pyplot as plt, numpy as np
from .criterion_criterion import Criterion, reduce
from ..modules.modules_bottleneck import MLP
from ..utils.misc import print_stats
import pdb


'''
def protected_logsum(tensor, dim):
    sorted_tensor, _ = torch.sort(tensor, dim=dim, descending=True)
    device = tensor.device
    dim_shape = tensor.shape[dim]; index = torch.arange(1, dim_shape).to(device)
    a_0 = torch.index_select(sorted_tensor, dim, torch.LongTensor([0]).to(device))
    return a_0 + (1 + torch.sum(torch.index_select(sorted_tensor, dim, index) - a_0, dim=dim).exp().sum(dim)).log()
'''

history = []

class InfoNCE(Criterion):

    def loss(self, model=None, out=None, out_negative=None, true_ids=None, epoch=None, n_preds=None, *args, **kwargs):
        prediction_out = out['prediction']
        z_predicted = prediction_out.get('out')
        z_real = out['z_enc'][-1][:, -z_predicted.shape[1]:]
        density_ratio = model.prediction_module.get_density_ratio(z_predicted, z_real)

        #print_stats(out['z_enc'][-1], 'encoded codes')
        #print_stats(cpc_contexts, 'cpcs')
        #print_stats(z_predicted, 'predicted codes')

        cpcs = 0
        for i in range(density_ratio.shape[0]):
            cpcs = cpcs + torch.sum(torch.diag(torch.nn.LogSoftmax(dim=1)(density_ratio[i])))
        cpcs = - cpcs / (z_predicted.shape[0] * z_predicted.shape[1])


        '''
        if kwargs.get('period', 'train'):
            dists = torch.mean(torch.sqrt((z_predicted - z_real).pow(2)), (0, -1)).detach().cpu().numpy()
            if len(history) < epoch+1:  
                history.append(0)
            history[-1] += dists
        if kwargs.get('plot'):
            h = np.array(history)
            fig = plt.figure()
            for i in range(h.shape[-1]):
                plt.plot(h[:, i], alpha=0.5, color="r")
            fig.savefig('cpc.pdf')
        '''

        # first column of ratios are positive example
        #print_stats(density_ratio[:,:,1:])
        #loss = reduce(torch.sum(-density_ratio[:,:,0]+torch.sum(density_ratio[:,:,1:].exp(), dim=2).log(), dim=1), reduction='mean')

        # trick avoiding overflow
        #m, _ = torch.max(density_ratio, 2)
        #loss = reduce(torch.mean(-density_ratio[:,:,0] + m + torch.sum((density_ratio[:,:,1:]-m.unsqueeze(-1)).exp(), 2).log(), dim=1), reduction='mean')

        #return loss, (loss,)
        return cpcs, (float(cpcs),)


    def get_named_losses(self, losses):
        return {'cpc': losses[0]}





    # def loss(self, out=None, **kwargs):
