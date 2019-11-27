import torch, numpy as np, os, gc, copy, pdb, time
from time import time, process_time
import matplotlib.pyplot as plt
from ..monitor.visualize_monitor import Monitor
from ..utils.dataloader import DataLoader
from ..utils.misc import GPULogger, sample_normalize, NaNError, print_module_grad, print_module_stats,checklist,  apply, apply_method
from .train_run import run, run_accumulate


def rec_attr(obj, attr, *args, **kwargs):
    assert type(attr) == str
    if issubclass(type(obj), list):
        return [rec_attr(o, attr) for o in obj]
    if issubclass(type(obj), tuple):
        return tuple([rec_attr(o, attr) for o in obj])
    else:
        return getattr(obj, attr)



class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError


def log():
    torch.cuda.synchronize()
    print(torch.cuda.memory_allocated(), torch.cuda.memory_cached())

def log_dist(msg, dist):
    with torch.no_grad():
        msg = "%s mean mean : %s \n --  mean var %s \n -- var mean %s \n -- var vars %s \n"%(msg, dist.mean.mean(0).squeeze(), dist.mean.std(0).squeeze(), dist.stddev.mean(0).squeeze(), dist.stddev.std(0).squeeze())
        msg += "\n%s mean min : %s \n --  mean max %s \n -- var min %s \n -- var max %s \n"%(msg, dist.mean.min(0)[0].squeeze(), dist.mean.max(0)[0].squeeze(), dist.stddev.min(0)[0].squeeze(), dist.stddev.max(0)[0].squeeze())
    return msg


class SimpleTrainer(Trainer):
    dataloader_class = DataLoader
    def __init__(self, models=None, datasets=None, losses=None, tasks=None, **kwargs):
        assert models is not None, "SimpleTrainer needs models"
        assert datasets is not None, "SimpleTrainer needs datasets"
        assert losses is not None, "SimpleTrainer needs losses"

        super(SimpleTrainer, self).__init__()
        self.name = kwargs.get('name', "model")
        self.models = models
        self.datasets = datasets
        self.losses = losses
        self.reinforcers = kwargs.get('reinforcers')
        self.tasks = kwargs.get('tasks', None)
        self.preprocessing = kwargs.get('preprocessing', None)
        self.dataloader_class = kwargs.get('dataloader', self.dataloader_class)
        # additional args
        self.trace_mode = kwargs.get('trace_mode', 'epoch')
        self.device = kwargs.get('device')
        self.loader_args = kwargs.get('loader_args', dict())
        self.split = kwargs.get('split', False)
        self.run = run_accumulate if self.split else run
        # plot args
        plot_tasks = kwargs.get('plot_tasks', tasks)
        self.plots = kwargs.get('plots', {})
        # init monitors
        self.init_monitors(models, datasets, losses, self.tasks, plots=self.plots, plot_tasks=plot_tasks)
        self.best_model = None
        self.best_test_loss = np.inf
        self.logger = GPULogger(kwargs.get('export_profile',None), kwargs.get('verbose', False))

    def init_monitors(self, models, datasets, losses, tasks, plots=None, plot_tasks=None, use_tensorboardx=False):
        self.monitor = Monitor(models, datasets, losses, tasks, plots=plots, tasks = plot_tasks, use_tensorboardx=use_tensorboardx)

    def init_time(self):
        self.start_time = process_time()

    def get_time(self):
        return process_time() - self.start_time

    def optimize(self, models, loss):
        #pdb.set_trace()
        apply_method(self.models, 'step', loss)
        apply_method(self.losses, 'step', loss)
        #print_grad_stats(self.models)

    def train(self, partition=None, write=False, batch_size=64, tasks=None, batch_cache_size=1, **kwargs):
        # set models and partition to train
        logger = GPULogger(verbose=True)
        apply_method(self.models, 'train')
        if partition is None:
            partition = 'train'
        # get dataloader
        loader = self.get_dataloader(self.datasets, batch_size=batch_size, partition=partition, tasks=tasks, batch_cache_size=batch_cache_size)
        # run
        full_loss, full_losses = self.run(self, loader, period="train", plot=False, **kwargs)
        # write losses in loss objects
        if self.trace_mode == "epoch" and write:
            if issubclass(type(self.losses), (list, tuple)):
                [apply_method(self.losses[i], 'write', 'train', full_losses[i], time=self.get_time()) for i in range(len(self.losses))]
            else:
                apply_method(self.losses, 'write', 'train', full_losses, time=self.get_time())
            self.monitor.update(kwargs.get('epoch'))
        return full_loss, full_losses

    def test(self, partition=None, write=False, tasks=None, batch_size=None, batch_cache_size=1, **kwargs):
        # set models and partition to test
        apply_method(self.models, 'eval')
        if partition is None:
            partition = 'test'
        # get dataloader and run without gradients
        with torch.no_grad():
            loader = self.get_dataloader(self.datasets, batch_size=batch_size, tasks=tasks, batch_cache_size=batch_cache_size, partition=partition)
            full_loss, full_losses = self.run(self, loader, optimize=False, schedule=True, period="test", plot=True, track_loss=True, **kwargs)
        # write losses in loss objects
        if write:
            if issubclass(type(self.losses), (list, tuple)):
                [apply_method(self.losses[i], 'write', 'test', full_losses[i], time=self.get_time()) for i in range(len(self.losses))]
            else:
                apply_method(self.losses, 'write', 'test', full_losses, time=self.get_time())
            self.monitor.update(kwargs.get('epoch'))
        # check if current model is the best model
        if full_loss < self.best_test_loss:
            self.best_model = self.get_best_model()
        return full_loss, full_losses

    def get_dataloader(self, datasets, batch_size=64, partition=None, tasks=None, batch_catch_size=1, **kwargs):
        return self.dataloader_class(datasets, preprocessing=self.preprocessing, batch_size=batch_size,
                                     tasks=self.tasks, batch_cache_size=1, partition=partition, **self.loader_args)

    def get_best_model(self, models=None, datasets=None, **kwargs):
        # get objects to save
        models = checklist(models or self.models)

        best_model = []
        for i,model in enumerate(models):
            current_device = next(model.parameters()).device
            # move best model to cpu to free GPU memory
            apply_method(model, 'cpu')
            if self.reinforcers:
                kwargs['reinforcers'] = self.reinforcers
            best_model.append(apply(apply_method(model, 'get_dict', **kwargs), copy.deepcopy))

            if current_device != torch.device('cpu'):
                with torch.cuda.device(current_device):
                    apply_method(model, 'cuda')
        return best_model

    def save(self, results_folder, models=None, save_best=True, **kwargs):
        # saving current model
        if models is None:
            models = self.models
        name = str(self.name)
        epoch = kwargs.get('epoch')
        print('-- saving model at %s'%'results/%s/%s.t7'%(results_folder, name))
        if not issubclass(type(models), list):
            models = [models]
        datasets = self.datasets
        if not issubclass(type(datasets), list):
            datasets = [datasets]
        partitions = rec_attr(datasets, "partitions")
        for i in range(len(models)):
            current_name = name if len(models) == 1 else '/vae_%d/%s'%(i, name)
            if not os.path.isdir(results_folder+'/vae_%d'%i):
                os.makedirs(results_folder+'/vae_%d'%i)
            if kwargs.get('epoch') is not None:
                current_name = current_name + '_' + str(kwargs['epoch'])
            additional_args = {'loss':self.losses, 'partitions':partitions, **kwargs}
            if self.reinforcers:
                additional_args['reinforcers'] = self.reinforcers
            models[i].save('%s/%s.t7'%(results_folder, current_name), **additional_args)

        # saving best model
        best_model = self.best_model
        if not issubclass(type(best_model), list):
            best_model = [best_model]
        if not self.best_model is None and save_best:
            print('-- saving best model at %s'%'results/%s/%s_best.t7'%(results_folder, name))
            for i in range(len(best_model)):
                current_name = name+'_best' if len(models) == 1 else '/vae_%d/%s_best'%(i, name)
                torch.save({'preprocessing':self.preprocessing, 'loss':self.losses, 'partitions':partitions, **kwargs, **best_model[i]}, '%s/%s.t7'%(results_folder, current_name))

    def plot(self, figures_folder, epoch=None, **kwargs):
        if epoch is not None:
            apply_method(self.monitor, 'update', epoch)
        plt.close('all')
        apply_method(self.monitor, 'plot', out=figures_folder, epoch=epoch, n_points=kwargs.get('plot_npoints'),
                          loader=self.dataloader_class,
                          preprocessing=self.preprocessing, trainer=self,
                          image_export=kwargs.get('image_export', False),
                          plot_reconstructions=kwargs.get('plot_reconstructions', False),
                          plot_latent=kwargs.get('plot_latentspace', False),
                          plot_statistics=kwargs.get('statistics', False), 
                          plot_partition=kwargs.get('plot_partition'),
                          sample=kwargs.get('sample', False))


def train_model(trainer,  options={}, save_with={}, **kwargs):
    epochs = options.get('epochs', 3000) # number of epochs
    save_epochs = options.get('save_epochs', 100) # frequency of model saving
    plot_epochs = options.get('plot_epochs', 100) # frequency of plotting
    batch_size = options.get('batch_size', 64) # batch size
    batch_split = options.get('batch_split')
    remote = options.get('remote', None) # automatic scp transfer

    # Setting results & plotting directories
    results_folder = options.get('results_folder', 'saves/' + trainer.name)
    figures_folder = options.get('figures_folder', results_folder+'/figures')
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)

    # Init training
    epoch = options.get('first_epoch', 0)
    trainer.init_time()
    while epoch < epochs:
        print('-- EPOCH %d'%epoch)
        # train
        train_loss, train_losses = trainer.train(epoch=epoch, batch_size=batch_size, batch_split=batch_split, write=True, **kwargs)
        print('final train loss : %f', train_loss)
        #print(' - losses : ', train_losses)

        # test
        with torch.no_grad():
            test_loss, test_losses = trainer.test(epoch=epoch, batch_size=batch_size, batch_split=batch_split, write=True, **kwargs)
            print('current test loss : %f', test_loss)
            #print(' - losses : ', test_losses)

        # save
        if epoch%save_epochs==0 or epoch == (epochs-1):
            trainer.save(results_folder, epoch=epoch, **save_with)

        # plot
        if epoch % plot_epochs == 0:
            trainer.plot(figures_folder, epoch=epoch, **options)
            if remote is not None:
                remote_folder= remote+'/figures_'+trainer.name
                print('scp -r %s %s:'%(figures_folder, remote_folder))
                os.system('scp -r %s %s:'%(figures_folder, remote_folder))


        # clean
        gc.collect(); gc.collect()

        # do it again
        epoch += 1


