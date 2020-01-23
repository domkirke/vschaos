#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:11:17 2018

@author: chemla
"""
import os, pdb, gc, time, random
import numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from librosa.core import load
from librosa.output import write_wav
from ..data.signal.transforms import  computeTransform, inverseTransform
from ..data import Dataset
from ..utils.trajectory import get_random_trajectory, get_interpolation
from ..utils import checklist, check_dir, choices



# CORE FUNCTIONS

def path2audio(model, current_z, transformOptions, n_interp=1, order_interp=1, from_layer=-1, out=None, preprocessing=None, graphs=True, norm=True, projection=None, **kwargs):
    # make interpolation
    if order_interp == 0:
        z_interp = np.zeros((current_z.shape[0]*n_interp, current_z.shape[1]))
        for i in range(current_z.shape[0]):
            z_interp[i*n_interp:(i+1)*n_interp] = current_z[i]
    else:
        coord_interp = np.linspace(0, current_z.shape[0]-1, (current_z.shape[0]-1)*n_interp)
        z_interp = np.zeros((len(coord_interp), current_z.shape[1]))
        for i,y in enumerate(coord_interp):
            z_interp[i] = ndimage.map_coordinates(current_z, [y * np.ones(current_z.shape[1]), np.arange(current_z.shape[1])], order=2)
        z_interp = torch.from_numpy(z_interp)

    # get corresponding sound distribution
    model.eval()
    with torch.no_grad():
        if model.take_sequences:
            z_interp = z_interp.unsqueeze(0)
        vae_out = model.decode( model.format_input_data(z_interp), from_layer=from_layer, sample=False )
    signal_out = vae_out[0]['out_params'].mean.squeeze()
    if len(signal_out.shape) > 2:
        signal_out = signal_out.reshape(signal_out.shape[0]*signal_out.shape[1], signal_out.shape[2])

    if model.take_sequences:
        z_interp = z_interp.squeeze(0)

    if graphs:
        if projection:
            fig = plt.figure()
            ax_latent = fig.add_subplot(121, projection="3d")
            z_projected = projection.transform(z_interp)
            if len(z_projected.shape) > 2:
                z_projected = z_projected[0]
            ax_latent.plot(z_projected[:,0], z_projected[:, 1], z_projected[:, 2])
            fig.add_subplot(122).imshow(signal_out.cpu().detach().numpy(), aspect='auto')
            fig.savefig(out+'.pdf', format="pdf")
        else:
            fig = plt.figure()
            plt.imshow(signal_out.cpu().detach().numpy(), aspect='auto')
            fig.savefig(out+'.pdf', format="pdf")

    # output signal
    if not preprocessing is None:
        signal_out = preprocessing.invert(signal_out)
    transform = transformOptions.get('transformType', 'stft')
    signal_out = inverseTransform(signal_out.cpu().detach().numpy(), transform, {'transformParameters':transformOptions}, **kwargs)
    write_wav(out+'.wav', signal_out, transformOptions.get('resampleTo', 22050), norm=norm)
    return z_interp



def window_signal(sig, window, overlap=None, axis=0, window_type=None, pad=True):
    overlap = overlap or window
    if pad:
        n_windows = sig.shape[axis] // overlap
        target_size = n_windows * overlap + window
        if sig.shape[axis] < target_size:
            pads = [(0,0)]*len(sig.shape)
            pads[axis]=(0,target_size-sig.shape[axis])
            sig = np.pad(sig, pads, mode='constant')
    else:
        n_windows = (sig.shape[axis]-window)//overlap
    sig = np.stack([sig[i*overlap:i*overlap+window] for i in range(n_windows)], axis=axis)
    if window_type:
        sig = sig * window_type(sig.shape[-1])
    return sig


def overlap_add(sig, axis=0, overlap=None, window_type=None, fusion="stack_right"):
    overlap = overlap or sig.shape[axis]
    new_size = sig.shape[axis]*overlap + sig.shape[axis+1]
    new_shape = (*sig.shape[:axis], new_size)
    if len(sig.shape) > axis:
        new_shape += sig.shape[axis+2:]
    sig_t = np.zeros(new_shape, dtype=sig.dtype)
    if window_type:
        sig *= window_type(sig.shape[-1])
    for i in range(sig.shape[axis]):
        idx = [slice(None)]*len(sig_t.shape); idx[axis] = slice(i*overlap, i*overlap+sig.shape[axis+1])
        idx_2 = [slice(None)]*len(sig.shape); idx_2[axis] = i
        print(sig.shape, sig_t.shape)
        print(idx_2, idx)
        if fusion == "stack_right":
            sig_t.__setitem__(idx, sig.__getitem__(idx_2))
        elif fusion == "overlap_add":
            sig_t.__getitem__(idx).__iadd__(sig.__getitem__(idx_2))
    return sig_t
        # sig_t.__getitem__((*take_all, slice(i*overlap,i*overlap+window))).__iadd__(sig.__getitem__()))


def get_transform_from_files(files, transform, transformOptions, window=None, take_sequences=False, merge_mode="min"):
    transform = transform or transformOptions.get('transformType')
    transform_datas = []
    for i, current_file in enumerate(files):
        # get transform from file
        if transform is not None:
            if issubclass(type(transform), (list, tuple)):
                transform = transform[0]
            current_transform = computeTransform([current_file], transform, transformOptions)[0]
            current_transform = np.array(current_transform)
        else:
            current_transform, _= load(current_file, sr=transformOptions.get('resampleTo', 22050))
        originalPhase = np.angle(current_transform)

        # window dat
        if window:
            overlap = overlap or window
            current_transform = window_signal(current_transform, window, overlap, axis=0)
        if take_sequences:
            current_transform = current_transform[np.newaxis]
        transform_datas.append(current_transform)
    if merge_mode == "min":
        if take_sequences:
            min_size = min([m.shape[1] for m in transform_datas])
            transform_datas = [td[:, :min_size] for td in transform_datas]
        else:
            min_size = min([m.shape[0] for m in transform_datas])
            transform_datas = [td[:min_size] for td in transform_datas]

    return transform_datas, originalPhase


# HIGH-LEVEL SYNTHESIS METHODS

def resynthesize_files(dataset, model, transformOptions=None, transform=None, preprocessing=None, out='./',
                       sample=False, iterations=50, export_original=True, method="griffin-lim", window=None,
                       overlap=None, norm=True, sequence=False, sequence_overlap=False, n_files=10, files=None,
                       predict=False, epoch=None, **kwargs):

    if files is None:
        if issubclass(type(dataset), Dataset):
            files = choices(dataset.files, k=n_files)
        else:
            files = choices(dataset, k=n_files)

    transform = transform or transformOptions.get('transformType')
    for i, current_file in enumerate(files):
        # get transform from file
        if transform is not None:
            if issubclass(type(transform), (list, tuple)):
                transform = transform[0]
            current_transform = computeTransform([current_file], transform, transformOptions)[0]
            current_transform = np.array(current_transform)
            # ct = np.copy(current_transform)
        else:
            current_transform, _= load(current_file, sr=transformOptions.get('resampleTo', 22050))
        originalPhase = np.angle(current_transform)
            # ct = np.copy(current_transform)
        path_out = out+'/audio_reconstruction/'+os.path.splitext(os.path.basename(current_file))[0]+('' if epoch is None else '_%d'%epoch)+'.wav'
        original_out = out+'/audio_reconstruction/'+os.path.splitext(os.path.basename(current_file))[0]+('' if epoch is None else '_%d'%epoch)+'_original.wav'

        # pre-processing
        print('synthesizing %s...'%path_out)
        if preprocessing:
            current_transform = preprocessing(current_transform)
        # window dat
        if window:
            overlap = overlap or window
            current_transform = window_signal(current_transform, window, overlap, axis=0)
        has_sequences = False
        if hasattr(model, 'take_sequences'):
            current_transform = current_transform[np.newaxis]
            has_sequences = True
        # add dimension if the model take conv inputs
        if model.pinput['conv']:
            conv_dim = - (len(checklist(model.pinput['dim'])) + 1)
            current_transform = np.expand_dims(current_transform, axis=conv_dim)
        # make sequences
        # if sequence:
        #     current_transform = window_signal(current_transform, sequence, sequence_overlap, axis=0)

        # forward
        t = time.process_time()
        with torch.no_grad():
            vae_out = model.forward(current_transform, **kwargs)
        transform_out = vae_out['x_params'].sample() if sample else vae_out['x_params'].mean

        # invert & export
        if model.pinput['conv']:
            current_transform = current_transform.squeeze(conv_dim)
            transform_out = transform_out.squeeze(conv_dim)
        if preprocessing:
            current_transform = preprocessing.invert(current_transform)
            transform_out = preprocessing.invert(transform_out)
        transform_out = transform_out.cpu().detach().numpy()

        if sequence:
            transform_out = overlap_add(transform_out, overlap=sequence_overlap)
            current_transform = overlap_add(current_transform, overlap=sequence_overlap)

        if window:
            transform_out = overlap_add(transform_out, overlap=overlap, window_type=np.hamming, fusion="overlap_add")
            current_transform = overlap_add(current_transform, overlap=overlap, window_type=np.hamming, fusion="overlap_add")

        if transform is not None:
            signal_out = inverseTransform(transform_out, transform, {'transformParameters':transformOptions}, originalPhase=originalPhase, iterations=iterations, method=method)
        else:
            signal_out = transform_out

        if not os.path.isdir(os.path.dirname(path_out)):
            os.makedirs(os.path.dirname(path_out))
        write_wav(path_out, signal_out, transformOptions.get('resampleTo', 22050), norm=norm)

        # export original in case
        if export_original:
            if transform:
                current_transform = inverseTransform(current_transform.squeeze(), transform, {'transformParameters':transformOptions},
                                                     originalPhase=originalPhase,iterations=iterations, method=method)
            write_wav(original_out, current_transform, transformOptions.get('resampleTo', 22050), norm=norm)

        del path_out; del signal_out; del current_transform
        gc.collect(); gc.collect();
        torch.cuda.empty_cache()

    return None, []


def trajectory2audio(model, traj_types, transformOptions, n_trajectories=1, n_steps=64, iterations=10, out=None,
                     preprocessing=None, projection=None, **kwargs):
    # load model
    if not os.path.isdir(out+'/trajectories'):
        os.makedirs(out+'/trajectories')
    paths = []
    for traj_type in traj_types:
        for l in range(len(model.platent)):
            # generate trajectories
            trajectories = get_random_trajectory(traj_type, model.platent[l]['dim'], n_trajectories, n_steps)
            # forward trajectory
            current_proj = projection
            if issubclass(type(projection), list):
                projection = projection[l]
            for i,t in enumerate(trajectories):
                z = path2audio(model, t, transformOptions, n_interp=1, preprocessing=preprocessing, out=out+'/trajectories/%s_%s_%s'%(traj_type, l,  i), iterations=iterations, from_layer=l, **kwargs)
            paths.append(z)
    return paths


def interpolate_files(dataset, vae, n_files=1, n_interp=10, out=None, preprocessing=None, preprocess=False,
                      projections=None, transformType=None, window=None, transformOptions=None, predict=False, **kwargs):
    for f in range(n_files):
        check_dir('%s/interpolations'%out)
        #sequence_length = loaded_data['script_args'].sequence
        #files_to_morph = random.choices(range(len(dataset.data)), k=2)
        files_to_morph = choices(dataset.files, k=2)

        data_outs = []
        projections = checklist(projections)
        with torch.no_grad():
            vae_input, original_phase = get_transform_from_files(files_to_morph, transformType, transformOptions, window=window, take_sequences=False, merge_mode="min")
            if preprocessing:
                vae_input = preprocessing(vae_input)
            vae_input = np.stack(vae_input, axis=0)
            if vae.pinput['conv']:
                conv_dim = - (len(checklist(vae.pinput['dim'])) + 1)
                vae_input = np.expand_dims(vae_input, axis=conv_dim)
            vae_input = vae.format_input_data(vae_input)
            z_encoded = vae.encode(vae_input,return_shifts=False)

            trajs = []
            for l in range(len(vae.platent)):
                # get trajectory between target points
                traj = get_interpolation(z_encoded[l]['out_params'].mean.cpu().detach(), n_steps=n_interp).float()
                #data_out = vae.decode([traj], n_steps=sequence_length)[0]['out_params'].mean
                device = next(vae.parameters()).device; traj = traj.to(device)
                #1data_outs.append(vae.decode([traj], predict=l==len(vae.platent)-1, from_layer=l)[0]['out_params'].mean)
                vae_out = vae.decode(traj, predict=l==len(vae.platent)-1, inplace=True, from_layer=l)[0]['out_params'].mean
                if preprocessing:
                    vae_out = preprocessing.invert(vae_out)
                data_outs.append(vae_out)
                trajs.append(traj)

        # plot things
        for l, data_out in enumerate(data_outs):
            if preprocess:
                data_out = preprocessing.invert(data_out)
            if projections:
                grid = gridspec.GridSpec(2,6)
                steps = np.linspace(0, n_interp-1, 6, dtype=np.int)
                fig = plt.figure(figsize=(12, 6))
                ax = fig.add_subplot(grid[:2, :2], projection='3d')
                proj = projections[l].transform(trajs[l].cpu().detach().numpy())
                cmap = plt.cm.get_cmap('plasma', n_interp)
                proj = proj.squeeze()
                if len(proj.shape) == 3:
                    for i in range(1, n_interp-1):
                        ax.plot(proj[i,:, 0], proj[i,:, 1], proj[i,:, 2], color=cmap(i), alpha=0.3)

                    ax.plot(proj[0, :, 0], proj[0, :, 1], proj[0, :, 2], color=cmap(0))
                    ax.scatter(proj[0, 0, 0], proj[0, 0, 1], proj[0, 0, 2], color=cmap(0), marker='o')
                    ax.scatter(proj[0, -1, 0], proj[0, -1, 1], proj[0, -1, 2], color=cmap(0), marker='+')
                    ax.plot(proj[-1,:, 0], proj[-1,:, 1], proj[-1,:, 2], color=cmap(n_interp))
                    ax.scatter(proj[-1,0, 0], proj[-1,0, 1], proj[-1,0, 2], color=cmap(n_interp), marker='o')
                    ax.scatter(proj[-1,-1, 0], proj[-1,-1, 1], proj[-1,-1, 2], color=cmap(n_interp), marker='+')
                for i in range(2):
                    for j in range(3):
                        ax = fig.add_subplot(grid[i,2+j])
                        spec = data_outs[l][steps[i*2+j]].squeeze().cpu().detach().numpy()
                        ax.imshow(spec, aspect="auto")
                        ax.set_xticks([]); ax.set_yticks([])

            else:
                raise NotImplementedError

            for i in range(data_out.shape[0]):
                signal_out = inverseTransform(data_out[i].squeeze().cpu().detach().numpy(), 'stft', {'transformParameters':transformOptions}, iterations=30, method='griffin-lim')
                write_wav('%s/interpolations/morph_%d_%d_%d.wav'%(out,l,f,i), signal_out, transformOptions.get('resampleTo', 22050), norm=True)

            fig.savefig('%s/interpolations/morph_%d_%d.pdf'%(out,l,f), format="pdf")
            plt.close('all')


def generate_autodrive(vae, dataset, n_files=1, out=None, preprocessing=None, transformOptions=None, start="random", n_start=10, n_loops=10, projections=None):
    for j in range(n_files):
        check_dir('%s/autodrive'%out)
        #sequence_length = loaded_data['script_args'].sequence
        # get starting point
        device = next(vae.parameters()).device
        if start == "file":
            input_file = random.randrange(len(dataset))
            data_in, _ = dataset[input_file]
            data_in = vae.format_input_data(preprocessing(data_in[:n_start])).unsqueeze(0)
            z_in = vae.encode(data_in)[-1]['out_params'].mean
        elif start == "random":
            latent_in = vae.platent[-1]['dim']
            # draw random point
            z0 = torch.distributions.Normal(torch.zeros(1, latent_in), torch.ones(1, latent_in)).sample()
            # draw direction
            u = torch.distributions.Normal(torch.zeros(1, latent_in), torch.ones(1, latent_in)).sample()
            increments = torch.linspace(0, 1e-1, n_start).unsqueeze(0)
            z_in = (z0 + increments.t() @ u).unsqueeze(0).to(device=device)

        with torch.no_grad():
            for n in range(n_loops):
                prediction_out = vae.prediction_module({'z_enc':[z_in]})
                z_in = torch.cat([z_in, prediction_out['out']], 1)
            data_out = vae.decode(z_in)[0]["out_params"].mean
        data_out = data_out.squeeze().cpu()

        # plot things
        fig = plt.figure()
        if len(data_out.shape) == 1:
            plt.plot(data_out)
        else:
            plt.imshow(data_out, aspect="auto")
        fig.savefig('%s/autodrive/drive_%d.pdf'%(out, j), format="pdf")
        plt.close('all')

        signal_out = inverseTransform(preprocessing.invert(data_out.squeeze().cpu().detach().numpy()), 'stft', {'transformParameters':transformOptions}, iterations=10, method='griffin-lim')
        write_wav('%s/autodrive/drive_%d.wav'%(out, j), signal_out, transformOptions.get('resampleTo', 22050), norm=True)
