import numpy as np, pdb, os
from . import SimpleTrainer
from .train_run import run_scan
from ..criterions.criterion_criterion import Criterion
from ..criterions.criterion_divergence import KLD
from ..utils.misc import apply_method, checklist
from ..utils.dataloader import MixtureLoader
from ..monitor.visualize_plotting import plot_tf_reconstructions, plot_losses, plot_mx_reconstructions, plot_mx_latent_space


class SCANTrainer(SimpleTrainer):
    transfer_loss = KLD
    def __init__(self, models=None, datasets=None, losses=None, tasks=None, symbols=None,  **kwargs):
        super().__init__(models, datasets, losses, tasks, **kwargs)
        self.run =run_scan
        self.symbols = symbols
        if kwargs.get('transfer_loss'):
            self.transfer_loss = kwargs['transfer_loss']
        if not issubclass(type(self.transfer_loss), Criterion):
            self.transfer_loss = self.transfer_loss()

    def get_dataloader(self, datasets, batch_size=64, partition=None, tasks=None, batch_catch_size=1, **kwargs):
        meta_tasks = tasks or self.tasks or []
        meta_tasks.extend(self.symbols)
        return self.dataloader_class(datasets[0], preprocessing=self.preprocessing, batch_size=batch_size,
                                     tasks=meta_tasks, partition=partition, **self.loader_args)

    def get_transfer_loss(self, out_audio, out_symbol):
        n_layers = len(out_audio['z_params_enc'])
        loss = 0.; losses = []
        l, ls = self.transfer_loss(out_audio['z_params_enc'][-1].squeeze(), out_symbol['z_params_enc'][-1])
        loss = loss + l; losses.extend(list(ls))
        return loss, tuple(losses)

    def plot(self, figures_folder, epoch=None, **kwargs):
        super().plot(figures_folder, epoch, **kwargs)
        if self.plots.get('reconstructions'):
            for p in self.datasets[0].partitions.keys():
                if not os.path.isdir(figures_folder+'/transfer'):
                    os.makedirs(figures_folder+'/transfer')
                out = figures_folder+'/transfer/transfer_%s_%d'%(p, epoch)
                rec_plot_args = dict(self.plots['reconstructions'])
                rec_plot_args['n_points'] = 5
                plot_tf_reconstructions(self.datasets, self.models, out=out, partition=p, preprocessing=self.preprocessing, **rec_plot_args)
                if not os.path.isdir(figures_folder+'/loss_transfer'):
                    os.makedirs(figures_folder+'/loss_transfer')
                out = figures_folder+'/loss_transfer'
                plot_losses(loss=self.transfer_loss, out=out)

    def train(self, partition='train', write=False, batch_size=64, tasks=None, batch_cache_size=1, **kwargs):
        full_loss, full_losses = super(SCANTrainer, self).train(partition=partition, write=write, batch_size=batch_size, tasks=tasks, **kwargs)
        if self.trace_mode == "epoch" and write:
            self.transfer_loss.write(partition, full_losses[2], time=self.get_time())
        return full_loss, full_losses


    def test(self, partition='test', write=False, tasks=None, batch_size=None, batch_cache_size=1, **kwargs):
        full_loss, full_losses= super(SCANTrainer, self).test(partition=partition, write=write, batch_size=batch_size, tasks=tasks, **kwargs)
        if self.trace_mode == "epoch" and write:
            self.transfer_loss.write(partition, full_losses[2], time=self.get_time())
        return full_loss, full_losses





"""
def add_extra_class(current_metadata, random_weights):
        current_metadata = np.concatenate((current_metadata, np.zeros((current_metadata.shape[0], 1))), 1)
        current_metadata[np.where(random_weights == 0)] = np.array([0.]*(current_metadata.shape[1]-1)+[1])
        return current_metadata

def format_symbol_data(meta_datasets,  current_ids, random_weights, zero_extra_class=False):
    if issubclass(type(meta_datasets), list):
        # recursively apply on different datasets
        symbols = []
        if not issubclass(type(zero_extra_class), list):
            zero_extra_class = [zero_extra_class]
        for i in range(len(meta_datasets)):
            symbols.extend(format_symbol_data(meta_datasets[i], current_ids[i], random_weights[:, i], zero_extra_class[i]))
        return symbols
    else:
        symbols = meta_datasets.data
        if not issubclass(type(symbols), list):
            symbols = [symbols]
        symbols = [md.copy()[current_ids] for md in symbols]
        if zero_extra_class:
            symbols = [add_extra_class(d, random_weights) for d in symbols]
    return symbols


class SCANTrainer(SimpleTrainer):
    dataloader_class = MixtureLoader

    def __init__(self, models=None, datasets=None, losses=None, zero_extra_class=False, random_mode=None, solo_models=None, **kwargs):
        super(SCANTrainer, self).__init__(models, datasets, losses, **kwargs)
        self.zero_extra_class = zero_extra_class
        self.solo_models = solo_models
        if random_mode is None:
            self.random_mode = 'bernoulli' if self.zero_extra_class else 'constant'

    def run(self, loader, preprocessing=None, epoch=None, optimize=True, **kwargs):
        batch = 0; current_loss = 0
        train_losses = None
        preprocessing = self.preprocessing or preprocessing
        if preprocessing is None:
            preprocessing = [None]*len(self.datasets)

        for mixture, x, y in loader:
            # retrieve audio data
            if not preprocessing[0] is None:
                mixture = preprocessing[0](mixture)

            # retrieve symbolic data
            symbols = format_symbol_data(self.datasets[1], loader.current_ids, loader.random_weights, zero_extra_class=self.zero_extra_class)
            if not preprocessing[1] is None:
                symbols = preprocessing[1](symbols)

            # forward
            sg_out = self.models[0].forward(mixture, y=y)
            if issubclass(type(self.models[1]), list):
                sm_out = [self.models[1][i].forward(symbols[i]) for i in range(len(self.models))]
            else:
                sm_out = self.models[1].forward(symbols)
            outs = [sg_out, sm_out]
            # outs = [apply_method(self.models[0], 'forward', mixture, y=y),
            #         [self.models[1][i].forward(symbols[i]) for i in range(len(self.models[1]))]]

            # compute loss
            random_weights = None
            if hasattr(loader, "random_weights"):
                random_weights = loader.random_weights
            batch_loss, losses = self.losses.loss(self.models, outs, target=[mixture, symbols], epoch=epoch, solos=x,
                                                  write='train', solo_models=self.solo_models, random_weights=random_weights)

            if optimize:
                self.optimize(self.models, batch_loss)

            print("epoch %d / batch %d / losses : %s "%(epoch, batch, self.losses.get_named_losses(losses)))
            current_loss += batch_loss
            train_losses = losses if train_losses is None else train_losses + losses

            # adversarial phase
            # if adversarial:
            #     affectations = torch.distributions.Bernoulli(torch.full((x1.shape[0],1), 0.5)).sample().to(current_device)
            #     adv_input = torch.where(affectations==1, x1, outs[0]['x_params'][0][0])
            #     adv_outs = adversarial(adv_input)
            #     adv_loss = nn.functional.binary_cross_entropy(adv_outs, affectations)
            #     print("-- adversarial loss : %f"%adv_loss)
            #     adv_loss.backward()
            #     adversarial.step(adv_loss)
            #     adversarial.update_optimizers({})
            #     adversarial_losses['train'].append(adv_loss.detach().cpu().numpy())

            batch += 1
            del outs[1]; del outs[0]
            del mixture; del symbols

        current_loss /= batch
        return current_loss, train_losses

    def optimize(self, models, batch_loss):
        models[0].step(batch_loss, retain_graph=True)
        models[1].step(batch_loss, retain_graph=True)
        models[0].update_optimizers({}); models[1].update_optimizers({})

    def get_dataloader(self, datasets, batch_size=64, partition=None, tasks=None, batch_cache_size=1, **kwargs):
        return self.dataloader_class(datasets[0], batch_size=batch_size, tasks=tasks, batch_cache_size=batch_cache_size, partition=partition, random_mode=self.random_mode)

    def plot(self, figures_folder, epoch=None, **kwargs):
        plot_tf_reconstructions(self.datasets[0], self.datasets[1], self.models, random_mode=self.random_mode, out=figures_folder+'/reconstructions_%d'%epoch, zero_extra_class=self.zero_extra_class, preprocessing=self.preprocessing)


class MixtureSCANTrainer(SCANTrainer):
    def get_best_model(self, models=None, datasets=None, **kwargs):
        models = models or self.models[0]
        datasets = datasets or self.datasets[0]
        super(MixtureSCANTrainer, self).get_best_model(models=models, datasets=datasets)

    def run(self, *args, **kwargs):
        res = super(MixtureSCANTrainer, self).run(*args, **kwargs)
        self.plot_mixture(*args, **kwargs)
        return res

    def optimize(self, models, batch):
        models[0].step(batch)

    def save(self, results_folder, models=None, save_best=True, **kwargs):
        models = models or self.models
        super(MixtureSCANTrainer, self).save(results_folder, models=models[0], save_best=True, **kwargs)

    def plot(self, figures_folder, epoch=None, **kwargs):
        self.current_fig[0].savefig(figures_folder+'/reconstruction_%d.pdf'%epoch)
        self.current_fig[1].savefig(figures_folder+'/latent_%d.pdf'%epoch)

    def plot_mixture(self, *args, **kwargs):
        solo_models = kwargs.get('solo_models')
        fig_reco, _ = plot_mx_reconstructions(self.datasets[0], self.models[0], solo_models=solo_models)
        fig_latent, _ = plot_mx_latent_space(self.datasets[0], self.models[0], tasks=self.tasks)
        self.current_fig = [fig_reco, fig_latent]
"""
