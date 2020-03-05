#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:03:10 2018

@author: chemla
"""
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numpy import fromstring, transpose
import torch, pdb, os
import numpy as np
import matplotlib.pyplot as plt
from . import visualize_plotting as lplt
from .audio_synthesize import resynthesize_files, interpolate_files
from .audio_descriptor import plot2Ddescriptors,plot3Ddescriptors
from ..utils import apply_method, checklist
from torch.utils.tensorboard import SummaryWriter
import collections



plot_hash = {'reconstructions': lplt.plot_reconstructions,
             'latent_space': lplt.plot_latent3,
             'latent_trajs': lplt.plot_latent_trajs,
             'latent_dims': lplt.plot_latent_dim,
             'latent_consistency': lplt.plot_latent_consistency,
             'statistics':lplt.plot_latent_stats,
             'images':lplt.image_export,
             'conv_weights': lplt.plot_conv_weights,
             'losses': lplt.plot_losses,
             'class_losses':lplt.plot_class_losses,
             'grid_latent':lplt.grid_latent,
             'descriptors_2d':plot2Ddescriptors,
             'descriptors_3d':plot3Ddescriptors,
             'audio_reconstructions': resynthesize_files,
             'audio_interpolate': interpolate_files}


def dict_merge(dct, merge_dct):

    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

class TensorboardHandler(object):
    def __init__(self, path="runs"):
        os.system(f'rm -rf {path}')
        self.writer = SummaryWriter(path)
        self._counter = 0
        self.models = []
        self.losses = []

    def update(self, models, losses, epoch=None):
        if epoch is None:
            epoch = self._counter
        apply_method(self, "add_model", models, epoch=epoch)
        apply_method(self, "add_loss", losses, epoch=epoch)
        self._counter += 1

    def add_model(self, model, epoch):
        # self.writer.add_graph(model, verbose=True)
        for n, p in model.named_parameters():
            self.writer.add_histogram('model/'+n,p.detach().cpu().numpy(),epoch)

    def add_loss(self, losses, epoch):
        for loss in losses:
            loss_names = list(loss.loss_history[list(loss.loss_history.keys())[0]].keys())
            for l in loss_names:
                # write last losses
                # pdb.set_trace()
                last_loss = {k: np.array(v[l]['values'][-1]) if len(v) > 0 else 0. for k, v in loss.loss_history.items()}
                print(l, epoch, last_loss)
                self.writer.add_scalars('loss_%s/'%l, last_loss, epoch)

    def add_image(self, name, pic, epoch):
        self.writer.add_image(name, pic, epoch)

    def add_audio(self, name, audio, sr):
        self.writer.add_audio(name, audio, sample_rate=sr)


class Monitor(object):
    def __init__(self, model, dataset, loss, labels, plots={}, synth={}, partitions=['train', 'test'],
                 output_folder=None, tasks=None, use_tensorboard=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.labels = labels
        self.tasks = tasks if tasks else None
        self.use_tensorboard = use_tensorboard
        self.output_folder = output_folder if output_folder else None
        self.plots = plots
        self.synth = synth
        self.partitions = partitions
        if use_tensorboard:
            self.writer = TensorboardHandler(use_tensorboard)
            
    def update(self, epoch=None):
        if self.use_tensorboard:
            self.writer.update(self.model, self.loss, epoch=epoch)

    def close(self):
        pass

    def record_image(self, image_list, name, epoch=None):
        if not self.use_tensorboard:
            return
        images = []
        for i, fig in enumerate(image_list):
            cv = FigureCanvas(fig)
            cv.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = fromstring(cv.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            image = np.transpose(image, (2,0,1)) / 255
            if self.writer:
                self.writer.add_image(name+'_'+str(i), image, epoch)


    def plot(self, out=None, epoch=None, loader=None, trainer=None, **kwargs):
        # plot reconstructions
        if out is None:
            out = self.output_folder
        plt.close('all')

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

    def synthesize(self, out=None, epoch=None, preprocessing=None, loader=None, trainer=None, **kwargs):
        # plot reconstructions
        if out is None:
            out = self.output_folder
        plt.close('all')

        for plot, plot_args in self.synth.items():
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
                        audio, sr = plot_hash[plot](dataset, self.model[i], loader=loader,
                                                    trainer=trainer, partition=partition, out=out, name=output_name, **plot_args)
                        if self.writer:
                            self.writer.add_audio(output_name, audio, sr)

                else:
                    output_name = None if out is None else '/%s_%s_%s'%(plot, partition, epoch)
                    for pa in plot_args:
                        losses = pa.get('loss')
                        reinforcers = pa.get('reinforcers')
                        current_name = output_name+'_'+pa.get('name', '')
                        if pa.get('name'):
                            current_name = current_name+'_'+pa['name']
                            del pa['name']
                        audio, sr = plot_hash[plot](self.dataset, self.model, loader=loader,# preprocessing=preprocessing,
                                                    trainer=trainer, partition=partition, name=current_name, out=out,epoch=epoch, **pa)
                        self.writer.add_audio(current_name, audio, sr)

        # if image_export:
        #     output_name = None if out is None else out+'/grid%s'%epoch_suffix
        #     fig, axes = lplt.image_export(self.dataset, self.model, self.labels, out=output_name, ids=reconstruction_ids)
        #     self.record_image(fig, 'images', epoch)

