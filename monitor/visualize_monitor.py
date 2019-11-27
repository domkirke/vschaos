#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:03:10 2018

@author: chemla
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import fromstring, transpose
import torch, pdb
import numpy as np
import matplotlib.pyplot as plt
from . import visualize_plotting as lplt
from .audio_descriptor import plot2Ddescriptors,plot3Ddescriptors
from ..utils import apply_method, checklist
import tensorboardX
import collections



plot_hash = {'reconstructions': lplt.plot_reconstructions,
             'latent_space': lplt.plot_latent3,
             'latent_trajs': lplt.plot_latent_trajs,
             'statistics':lplt.plot_latent_stats,
             'images':lplt.image_export,
             'conv_weights': lplt.plot_conv_weights,
             'losses': lplt.plot_losses,
             'class_losses':lplt.plot_class_losses,
             'descriptors_2d':plot2Ddescriptors,
             'descriptors_3d':plot3Ddescriptors}


def dict_merge(dct, merge_dct):

    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

class TensorboardXHandler(object):
    def __init__(self, path="runs"):
        self.writer = tensorboardX.SummaryWriter(path)
        self._counter = 0

    def update(self, models, losses, epoch=None):
        if epoch is None:
            epoch = self._counter
        apply_method(self, "add_model", models, epoch=epoch)
        apply_method(self, "add_loss", losses, epoch=epoch)
        self._counter += 1

    def add_model(self, model, epoch):
        for n, p in model.named_parameters():
            self.writer.add_histogram('model/'+n,p.detach().cpu().numpy(),epoch)

    def add_loss(self, losses, epoch):
        registers = losses.loss_history.keys()
        full_losses = {}
        for register in losses.loss_history.keys():
            # write last losses
            last_losses = {k: np.array(v[-1]) if len(v) > 0 else 0. for k, v in losses.loss_history[register].items()}
            self.writer.add_scalars('loss_%s/'%register, last_losses, epoch)
            # for k in losses.loss_history[register].keys():
            #     if not k in full_losses.keys():
            #         full_losses[k] = np.expand_dims(losses.loss_history[register][k],0)
            #     else:
            #         full_losses[k] = np.concatenate([full_losses[k],losses.loss_history[register][k][np.new_axis]], 0)

        # for k in full_losses.keys():
        #     self.writer.add_pr_curve('loss/%s/'%k, full_losses[k])




    def add_image(self, name, pic, epoch):
        self.writer.add_image(name, pic, epoch)


class Monitor(object):
    def __init__(self, model, dataset, loss, labels, plots={}, partitions=['train', 'test'], output_folder=None, tasks=None, use_tensorboardx=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.labels = labels
        self.tasks = tasks if tasks else None
        self.use_tensorboardx = use_tensorboardx
        self.output_folder = output_folder if output_folder else None
        self.plots = plots
        self.partitions = partitions
        if use_tensorboardx:
#            self.tx_folder = output_folder+'/runs' if output_folder else 'runs'
#            os.system('rm -rf %s'%self.tx_folder)
#            os.system('tensorboard --logdir %s &> /dev/null &'%self.tx_folder)
#            os.system('echo $! > %s/.tbx_id'%self.tx_folder)
            self.writer = TensorboardXHandler(use_tensorboardx)
            
    def update(self, epoch=None):
        if self.use_tensorboardx:
            self.writer.update(self.model, self.loss, epoch=epoch)

    def close(self):
        pass
#        if self.use_tensorboardx:
#            os.system('cat %s |xargs kill'%self.tx_folder)

    def record_image(self, image_list, name, epoch=None):
        if not self.use_tensorboardx:
            return
        images = []
        for i, fig in enumerate(image_list):
            cv = FigureCanvas(fig)
            cv.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = fromstring(cv.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            image = np.transpose(image, (2,0,1)) / 255
            self.writer.add_image(name+'_'+str(i), image, epoch)


    def plot(self, out=None, epoch=None, preprocessing=None, loader=None, trainer=None, **kwargs):
        # plot reconstructions
        if out is None:
            out = self.output_folder
        plt.close('all')

        with torch.no_grad():
            for plot, plot_args in self.plots.items():
                plot_args = checklist(plot_args)
                plot_args = [dict(p) for p in plot_args]
                print('--monitor : %s, %s'%(plot, plot_args))

                for partition in self.partitions:
                    if issubclass(type(self.model), (list, tuple)):
                        for i in range(len(self.model)):
                            output_name = None if out is None else '/%s_%s_%s_%s'%(plot, partition, epoch, i)
                            dataset = self.dataset if not issubclass(type(self.dataset), (list, tuple)) else self.dataset[i]
                            if issubclass(type(losses), (list, tuple)):
                                plot_args['loss'] = losses[i]
                            if issubclass(type(reinforcers), (list, tuple)):
                                plot_args['reinforcers'] = reinforcers[i]

                            fig, axes = plot_hash[plot](dataset, self.model[i], loader=loader,
                                                        trainer=trainer, partition=partition, out=out, name=output_name, **plot_args)
                            self.record_image(fig, '%s_%s_%s'%(plot, partition, i), epoch)

                    else:
                        output_name = None if out is None else '/%s_%s_%s'%(plot, partition, epoch)

                        for pa in plot_args:
                            losses = pa.get('loss')
                            reinforcers = pa.get('reinforcers')
                            current_name = output_name+'_'+pa.get('name', '')
                            if pa.get('name'):
                                current_name = current_name+'_'+pa['name']
                                del pa['name']
                            fig, axes = plot_hash[plot](self.dataset, self.model, loader=loader,# preprocessing=preprocessing,
                                                        trainer=trainer, partition=partition, name=current_name, out=out,epoch=epoch, **pa)
                            self.record_image(fig, '%s_%s_%s'%(plot, out+'/'+current_name, partition), epoch)

        # if image_export:
        #     output_name = None if out is None else out+'/grid%s'%epoch_suffix
        #     fig, axes = lplt.image_export(self.dataset, self.model, self.labels, out=output_name, ids=reconstruction_ids)
        #     self.record_image(fig, 'images', epoch)

