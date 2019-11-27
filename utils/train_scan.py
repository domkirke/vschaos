#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:54:24 2018

@author: chemla
"""
import torch, os, pdb, copy, gc, numpy as np
import torch.nn as nn
from vschaos.utils.dataloader import DataLoader, MixtureLoader
from vschaos.monitor import visualize_plotting as lplt
from vschaos.monitor.visualize_dimred import PCA
from vschaos.monitor.visualize_monitor import Monitor
from torchvision.utils import save_image


def update_losses(losses_dict, new_losses):
    for k, v in new_losses.items():
        if not k in losses_dict.keys():
            losses_dict[k] = []
        losses_dict[k].append(new_losses[k])
    return losses_dict


def train_model(datasets, meta_datasets, models, loss, adversarial=None, tasks=None, loss_tasks=None, preprocessing=[None,None], options={}, plot_options={}, save_with={}):
    # Global training parameters
    name = options.get('name', 'model')
    epochs = options.get('epochs', 10000)
    save_epochs = options.get('save_epochs', 100)
    plot_epochs = options.get('plot_epochs', 100)
    batch_size = options.get('batch_size', 64)
    nb_reconstructions = options.get('nb_reconstructions', 3)
    remote = options.get('remote', None)
    random_mode = options.get('random_mode', 'constant')
    zero_extra_class = options.get('zero_extra_class', 0)
    if loss_tasks is None:
        loss_tasks = tasks if not tasks is None else None

    current_device = next(models[0].parameters()).device
    # Setting results & plotting directories
    results_folder = options.get('results_folder', 'saves/'+name)
    figures_folder = options.get('figures_folder', results_folder+'/figures')
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)
    if not os.path.isdir(results_folder+'/vae_1'):
        os.makedirs(results_folder+'/vae_1')
    if not os.path.isdir(results_folder+'/vae_2'):
        os.makedirs(results_folder+'/vae_2')

    # Init training
    epoch = options.get('first_epoch') or 0
    min_test_loss = np.inf; best_model = None

    reconstruction_ids = [np.random.permutation(len(datasets[0]))[:nb_reconstructions**2]]
    if adversarial:
        adversarial_losses = {'train':[], 'test':[]}

    # Start training!
    while epoch < epochs:
        print('-----EPOCH %d'%epoch)
        loader = MixtureLoader(datasets, batch_size=batch_size, partition='train', tasks=tasks, random_mode = random_mode)

        # train phase
        batch = 0; current_loss = 0;
        train_losses = None
        models[0].train(); models[1].train()
        for mixture, x, y in loader:
            # format x1
            if not preprocessing[0] is None:
                mixture = preprocessing[0](mixture)

            # format x2
            symbols = []
            for d in range(len(meta_datasets)):
                if issubclass(type(meta_datasets[d].data), list):
                    current_metadata = [x_tmp[loader.current_ids[d]] for x_tmp in meta_datasets[d].data]
                    if zero_extra_class:
                        for i_tmp, tmp in enumerate(current_metadata):
                            current_metadata[i_tmp] = np.concatenate((tmp, np.zeros((tmp.shape[0], 1))), 1)
                            current_metadata[i_tmp][np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                    symbols.extend(current_metadata)
                else:
                    current_metadata = meta_datasets[d].data[loader.current_ids[d]]
                    if zero_extra_class:
                        current_metadata = np.concatenate((current_metadata, np.zeros((current_metadata.shape[0], 1))), 1)
                        current_metadata[np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                    symbols.append(current_metadata)

            # forward
            outs = [models[0].forward(mixture, y=y), models[1].forward(symbols, y=y)]
            batch_loss, losses = loss.loss(models, outs, target=[mixture, symbols], epoch=epoch, write='train')
            if train_losses is None:
                train_losses = losses
            else:
                train_losses += losses

            models[0].step(batch_loss, retain_graph=True)
            models[1].step(batch_loss, retain_graph=True)

            print("epoch %d / batch %d / losses : %s "%(epoch, batch, loss.get_named_losses(losses)))
            current_loss += batch_loss
            models[0].update_optimizers({}); models[1].update_optimizers({})

            # adversarial phase
            if adversarial:
                affectations = torch.distributions.Bernoulli(torch.full((x1.shape[0],1), 0.5)).sample().to(current_device)
                adv_input = torch.where(affectations==1, x1, outs[0]['x_params'][0][0])
                adv_outs = adversarial(adv_input)
                adv_loss = nn.functional.binary_cross_entropy(adv_outs, affectations)
                print("-- adversarial loss : %f"%adv_loss)
                adv_loss.backward()
                adversarial.step(adv_loss)
                adversarial.update_optimizers({})
                adversarial_losses['train'].append(adv_loss.detach().cpu().numpy())

            batch += 1
            del outs[1]; del outs[0];
            del mixture; del symbols;
            torch.cuda.empty_cache()


        current_loss /= batch
        print('--- FINAL LOSS : %s'%current_loss)
        loss.write('train', train_losses)

        if torch.cuda.is_available():
            if current_device != 'cpu':
                if epoch ==0:
                    os.system('echo "epoch %d : allocated %d cache %d \n" > %s/gpulog.txt'%(epoch, torch.cuda.memory_allocated(current_device.index), torch.cuda.memory_cached(current_device.index), results_folder))
                else:
                    os.system('echo "epoch %d : allocated %d cache %d \n" >> %s/gpulog.txt'%(epoch, torch.cuda.memory_allocated(current_device.index), torch.cuda.memory_cached(current_device.index), results_folder))


        ## test_phase
        n_batches = 0
        with torch.no_grad():
            models[0].eval(); models[1].eval()
            loader = MixtureLoader(datasets, batch_size=None, partition='test', tasks=tasks, random_mode = random_mode)

            test_loss = None;
            if adversarial:
                adv_loss = torch.tensor(0., device=next(adversarial.parameters()).device)
            # train phase
            for mixture, x, y in loader:
                # format x1
                n_batches += 1
                if not preprocessing[0] is None:
                    mixture = preprocessing[0](mixture)

                # format x2
                symbols = []
                for d in range(len(meta_datasets)):
                    if issubclass(type(meta_datasets[d].data), list):
                        current_metadata = [x_tmp[loader.current_ids[d]] for x_tmp in meta_datasets[d].data]
                        if zero_extra_class:
                            for i_tmp, tmp in enumerate(current_metadata):
                                current_metadata[i_tmp] = np.concatenate((tmp, np.zeros((tmp.shape[0], 1))), 1)
                                current_metadata[i_tmp][np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                        symbols.extend(current_metadata)
                    else:
                        current_metadata = meta_datasets[d].data[loader.current_ids[d]]
                        if zero_extra_class:
                            current_metadata = np.concatenate((current_metadata, np.zeros((current_metadata.shape[0], 1))), 1)
                            current_metadata[np.where(loader.random_weights[:, d] == 0)] = np.array([0.]*(current_metadata[i_tmp].shape[1]-1)+[1])
                        symbols.append(current_metadata)

                # forward
                outs = [models[0].forward(mixture, y=y), models[1].forward(symbols, y=y)]
                current_test_loss, losses = loss.loss(models, outs, target=[mixture, symbols],epoch=epoch, write='test')
                if not test_loss:
                    test_loss = current_test_loss
                else:
                    test_loss = test_loss + current_test_loss

                if adversarial:
                    adv_in = torch.cat((x1, outs[0]['x_params'][0]), 0)
                    adv_out = adversarial(adv_in)
                    adv_target = torch.cat((torch.ones((x1.shape[0], 1), device=current_device), torch.zeros((x1.shape[0], 1), device=current_device)), 0)
                    adv_loss = adv_loss + nn.functional.binary_cross_entropy(adv_out, adv_target)


                del mixture; del symbols; del outs;
                gc.collect(); gc.collect()
                torch.cuda.empty_cache()

            print('test loss : ', test_loss / n_batches)
            if adversarial:
                print('adversarial loss : ', adv_loss)
                adversarial_losses['train'].append(adv_loss.detach().cpu().numpy())

            # register model if best test loss
            if current_test_loss < min_test_loss:
                min_test_loss = current_test_loss
                print('-- best model found at epoch %d !!'%epoch)
                if torch.cuda.is_available():
                    with torch.cuda.device_of(next(models[0].parameters())):
                        models[0].cpu(); models[1].cpu()
                        best_model = [copy.deepcopy(models[0].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets])),
                                      copy.deepcopy(models[1].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets]))]
                        models[0].cuda(); models[1].cuda()
                else:
                    best_model = [copy.deepcopy(models[0].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets])),
                                  copy.deepcopy(models[1].get_dict(loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets]))]
            gc.collect(); gc.collect()
            torch.cuda.empty_cache()

            # schedule training
            models[0].schedule(test_loss); models[1].schedule(test_loss)

            if epoch % plot_epochs == 0:
                lplt.plot_tf_reconstructions(datasets, meta_datasets, models, random_mode = random_mode, out=figures_folder+'/reconstructions_%d'%epoch, zero_extra_class=zero_extra_class, preprocessing=preprocessing)

            if epoch % save_epochs == 0:
                models[0].save(results_folder + '/vae_1/%s_%d.t7'%(name,epoch), loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets], **save_with)
                models[1].save(results_folder + '/vae_2/%s_%d.t7'%(name,epoch), loss=loss, epoch=epoch, partitions=[d.partitions for d in datasets], **save_with)
                if best_model:
                    torch.save({**save_with, **best_model[0]}, results_folder + '/vae_1/%s_best.t7'%(name))
                    torch.save({**save_with, **best_model[1]}, results_folder + '/vae_2/%s_best.t7'%(name))
                if adversarial:
                    torch.save(adversarial_losses, results_folder+'adversarial_log.t7')

        epoch += 1
