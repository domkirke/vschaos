#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:54:24 2018

@author: chemla
"""
import pdb
#pdb.set_trace()
import torch, os, pdb, copy, gc
from .dataloader import DataLoader

from ..monitor import visualize_plotting as lplt
from ..monitor.visualize_dimred import PCA
from ..monitor.visualize_monitor import Monitor

from torchvision.utils import save_image
import numpy as np

import matplotlib.pyplot as plt
from time import time



def update_losses(losses_dict, new_losses):
    for k, v in new_losses.items():
        if not k in losses_dict.keys():
            losses_dict[k] = []
        losses_dict[k].append(new_losses[k])
    return losses_dict


def train_model(dataset, model, loss, tasks=None, loss_tasks=None, preprocessing = None, options={}, plot_options={}, save_with={}):
    # Global training parameters
    name = options.get('name', 'model')
    epochs = options.get('epochs', 3000)
    save_epochs = options.get('save_epochs', 100)
    plot_epochs = options.get('plot_epochs', 100)
    batch_size = options.get('batch_size', 64)
    nb_reconstructions = options.get('nb_reconstructions', 3)
    remote = options.get('remote', None)
    batch_cache_size = options.get('offline_cache', 1)
    if loss_tasks is None:
        loss_tasks = tasks if not tasks is None else None
    
    # Setting results & plotting directories
    results_folder = options.get('results_folder', 'saves/'+name)
    figures_folder = options.get('figures_folder', results_folder+'/figures')
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)
        
    # Init training
    epoch = options.get('first_epoch') or 0
    min_test_loss = np.inf; best_model = None
    reconstruction_ids = np.random.permutation(len(dataset))[:nb_reconstructions**2]
    
    plot_tasks = plot_options.get('plot_tasks', dataset.tasks)
    use_tensorboardx = results_folder if plot_options.get('use_tensorboardx', False) else None
    monitor = Monitor(model, dataset, loss, tasks, tasks=plot_tasks, output_folder=results_folder, use_tensorboardx = use_tensorboardx)


    # Start training!
    while epoch < epochs:
        print('-----EPOCH %d'%epoch)
        loader = DataLoader(dataset, batch_size=batch_size, partition='train', tasks=tasks, batch_cache_size=batch_cache_size)
        
        # train phase
        batch = 0; current_loss = 0;
        train_losses = None
        model.train(); loss.train()
        for x,y in loader:
            # forward
            if not preprocessing is None:
                x = preprocessing(x)
            x = model.format_input_data(x, requires_grad=False)
            out = model.forward(x, y=y)
            
            # compute loss
            batch_loss, losses = loss.loss(model=model, out=out, target=x, epoch=epoch, write='train')
            train_losses = losses if train_losses is None else train_losses + losses 

            # learn
            model.step(batch_loss)
#            model.update_optimizers({})
            loss.step()
            
            # update loop
            print("epoch %d / batch %d / losses : %s "%(epoch, batch, loss.get_named_losses(losses)))
            current_loss += batch_loss
            batch += 1
            del out; del x;
            torch.cuda.empty_cache()
            
        current_loss /= batch
        print('--- FINAL LOSS : %s'%current_loss)
        loss.write('train', train_losses)

        # make gpu log at each epoch
        if torch.cuda.is_available():
            current_device = next(model.parameters()).device
            if current_device != 'cpu':
                if epoch ==0:
                    os.system('echo "epoch %d : allocated %d cache %d \n" > %s/gpulog.txt'%(epoch, torch.cuda.memory_allocated(current_device.index), torch.cuda.memory_cached(current_device.index), results_folder)) 
                else:
                    os.system('echo "epoch %d : allocated %d cache %d \n" >> %s/gpulog.txt'%(epoch, torch.cuda.memory_allocated(current_device.index), torch.cuda.memory_cached(current_device.index), results_folder)) 


        # test_phase
        with torch.no_grad():
            model.eval(); loss.eval()
            loader = DataLoader(dataset, batch_size=None, partition='test', task=tasks, batch_cache_size=batch_cache_size)
            test_losses = None
            current_test_loss = 0
            for x,y in loader:
                test_data = model.format_input_data(x)
                out = model.forward(test_data)
                test_loss, losses = loss.loss(model=model, out=out, target=test_data, epoch=epoch, write='test')
                current_test_loss += test_loss
                if test_losses is None:
                    test_losses = losses 
                else:
                    test_losses += losses 
                del test_data; del out;
                gc.collect(); gc.collect()
                torch.cuda.empty_cache()
            print('test loss : ', current_test_loss)
            
            # register model if best test loss
            if current_test_loss < min_test_loss:
                min_test_loss = current_test_loss
                print('-- best model found at epoch %d !!'%epoch)
                if torch.cuda.is_available():
                    with torch.cuda.device_of(next(model.parameters())):
                        model.cpu()
                        best_model = copy.deepcopy(model.get_dict(loss=loss, epoch=epoch, partitions=dataset.partitions))
                        model.cuda()
                else:
                    best_model = copy.deepcopy(model.get_dict(loss=loss, epoch=epoch, partitions=dataset.partitions))
            gc.collect(); gc.collect()
            torch.cuda.empty_cache()
                    
            # schedule training
            model.schedule(test_loss)
         
        
        plt.ioff()
        
        monitor.update(epoch)
        # Save models
#        if epoch%save_epochs==0 or epoch == (epochs-1):
#            print('-- saving model at %s'%'results/%s/%s_%d.t7'%(results_folder, name, epoch))
#            model.save('%s/%s_%d.t7'%(results_folder, name, epoch), loss=loss, epoch=epoch, partitions=dataset.partitions, **save_with)
#            if not best_model is None:
#                print('-- saving best model at %s'%'results/%s/%s_best.t7'%(results_folder, name))
#                torch.save({**save_with, **best_model}, '%s/%s_best.t7'%(results_folder, name))
                
        # Make plots
        if plot_options.get('image_export', False):
            monitor.plot(out=figures_folder, epoch=epoch, image_export=True, reconstruction_ids=reconstruction_ids) # image export made at every epoch
            
        if epoch%plot_epochs == 0:
            plt.close('all')
            n_points = plot_options.get('plot_npoints', min(dataset.data.shape[0], 5000))
            monitor.plot(out=figures_folder, epoch=epoch, n_points=n_points, preprocessing=plot_options.get('preprocessing'),
                         plot_reconstructions=plot_options.get('plot_reconstructions', False),
                         plot_latent = plot_options.get('plot_latentspace', False),
                         plot_statistics = plot_options.get('statistics', False))
            if torch.cuda.is_available():
                gc.collect(); gc.collect()
                torch.cuda.empty_cache()
#            if plot_options.get('plot_distributions', False):
#                print('plotting latent distributions...')
#                lplt.plot_latent_dists(dataset, model, label=tasks, tasks=plot_tasks, n_points=n_points, out=figures_folder+'/dists_%d'%epoch, 
#                                       dims=plot_dimensions,split=False, legend=True, bins=10, relief=True)
#            if plot_options.get('plot_losses', False):
#                print('plotting losses...')
#                lplt.plot_class_losses(dataset, model, loss, label=tasks, tasks=plot_tasks, loss_task = tasks, out=figures_folder+'/losses')

            if not remote is None:
                remote_folder= remote+'/figures_'+name
                print('scp -r %s %s:'%(figures_folder, remote_folder))
                os.system('scp -r %s %s:'%(figures_folder, remote_folder))
            
        epoch += 1

    monitor.close()


        
