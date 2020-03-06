import torch, os, pdb, gc
import numpy as np
import matplotlib.pyplot as plt
import librosa
from functools import reduce
from torchvision.utils import make_grid

from mpl_toolkits.mplot3d import Axes3D
from . import visualize_dimred as dr
from ..utils.onehot import fromOneHot, oneHot
from ..utils import decudify, merge_dicts, CollapsedIds, check_dir, recgetitem, decudify
from ..utils.dataloader import DataLoader
from ..utils import get_latent_out, get_flatten_meshgrid
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.ticker import StrMethodFormatter
from torchvision.utils import make_grid, save_image

from . import visualize_core as core
from ..modules.modules_convolution import *


################################################
########        RECONSTRUCTION PLOTS
####

eps=1e-7
                
def plot_reconstructions(dataset, model, label=None, n_points=10, out=None, preprocess=True, preprocessing=None, partition=None,
                         epoch=None, name=None, loader=None, ids=None, plot_multihead=False, reinforcers=None, **kwargs):

    # get plotting ids
    if partition is not None:
        dataset = dataset.retrieve(partition)
    n_rows, n_columns = core.get_divs(n_points)
    if ids is None:
        full_id_list = np.arange(len(dataset))
        ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids

    # get data
    Loader = loader if loader else DataLoader
    loader = Loader(dataset, None, ids=ids, is_sequence=model.take_sequences, tasks=label)

    data, metadata = next(loader.__iter__())
    if preprocess:
        if issubclass(type(model.pinput), list):
            preprocessing = preprocessing if preprocessing else [None]*len(dataset.data)
            data_pp = [None]*len(dataset.data)
            if not issubclass(type(preprocessing), list):
                preprocessing = [preprocessing]*len(dataset.data)
            for i, pp in enumerate(preprocessing):
                if not pp is None:
                    data_pp[i] = preprocessing(data[i])
        else:
            data_pp = preprocessing(data) if preprocessing is not None else data
    else:
        data_pp = data

    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    add_args['epoch'] = kwargs.get('epoch')

    with torch.no_grad():
        vae_out = model.forward(data_pp, y=metadata, **add_args)
    if reinforcers is not None:
        vae_out = reinforcers.forward(vae_out)

    vae_out['x_params'] = checklist(vae_out['x_params'])
    if vae_out.get('x_reinforced') is not None:
        vae_out['x_reinforced'] = checklist(vae_out['x_reinforced'])
    data_pp = checklist(data_pp)

    figs = {}; axes = {}
    if out:
        out += '/reconstruction/'
        check_dir(out)

    suffix = ""
    suffix = "_"+str(partition) if partition is not None else suffix
    suffix = suffix+"_%d"%epoch if epoch is not None else suffix
    for i in range(len(vae_out['x_params'])):
        multihead_outputs = None
        if plot_multihead and hasattr(model.decoders[0].out_modules[0], 'current_outs'):
            multihead_outputs = model.decoders[0].out_modules[0].current_outs

        if vae_out.get('x_params') is not None:
            fig_path = name if len(vae_out) == 1 else '%s_%d'%(name, i)
            fig, ax = core.plot_distribution(vae_out['x_params'][i], target=data_pp[i], preprocessing=preprocessing, preprocess=preprocess, multihead=multihead_outputs, out=fig_path, **kwargs)
            figs[os.path.basename(fig_path)] = fig; axes[os.path.basename(fig_path)] = ax
            if not out is None:
                fig_path = f"{out}/{fig_path}{suffix}.pdf"
                fig.savefig(fig_path, format="pdf")

        fig_reinforced = None
        if vae_out.get('x_reinforced') is not None:
            fig_reinforced, ax_reinforced = core.plot_distribution(vae_out['x_reinforced'][i], target=data_pp[i], preprocessing=preprocessing, preprocess=preprocess, out=fig_path, multihead=multihead_outputs, **kwargs)
            fig_path = out+'_%s_reinforced'%name if len(vae_out) == 1 else out+'%s_%d_reinforced'%(name, i)
            figs[os.path.basename(fig_path)+'_reinforced'] = fig; axes[os.path.basename(fig_path)+'_reinforced'] = ax
            if not out is None:
                fig_path = f"{out}/{name}{suffix}.pdf"
                fig.savefig(fig_path, format="pdf")

    del data; del vae_out
    return figs, axes


def get_plot_subdataset(dataset, n_points=None, partition=None, ids=None):
    if ids is None:
        if partition:
            current_partition = dataset.partitions[partition]
            ids = current_partition[np.random.permutation(len(current_partition))[:n_points]]
        else:
            ids = np.random.permutation(dataset.data.shape[0])[:n_points]
    dataset = dataset.retrieve(ids)
    return dataset, ids


def grid_latent(dataset, model, layer=-1, reduction=dr.PCA, n_points=None, ids=None, grid_shape=10, scales=[-3.0, 3.0], batch_size=None, loader=None, label=None, out=None, epoch=None, partition=None, **kwargs):
    n_dims = 2 #n_dims or model.platent[layer]['dim']
    zs, meshgrids, idxs = get_flatten_meshgrid(n_dims, scales, grid_shape); 
    idxs_hash = {i:idxs[i] for i in range(len(idxs))}

    latent_dims = model.platent[layer]['dim']
    if latent_dims > 2:
        dataset = get_plot_subdataset(dataset, n_points=n_points, partition=partition, ids=ids)
        Loader = loader if loader else DataLoader
        loader = Loader(dataset[0], batch_size, is_sequence=model.take_sequences, tasks=label)
        outs = []
        with torch.no_grad():
            for x, y in loader:
                outs.append(model.encode(model.format_input_data(x)))
        outs = merge_dicts(outs) 
        #TODO general sampling
        data_zs = outs[layer]['out_params'].mean
        dimred = reduction(n_components=n_dims)
        dimred.fit_transform(data_zs.cpu().numpy())
        zs = dimred.inverse_transform(zs)

    with torch.no_grad():
        outs = model.decode(model.format_input_data(zs), layer=layer)[0]['out_params']
    
    grid = torch.zeros(*meshgrids[0].shape, 1, *outs.mean.shape[1:])
    grid_std = torch.zeros(*meshgrids[0].shape, 1, *outs.mean.shape[1:])

    for n, idx in enumerate(idxs):
        x, y = idx
        grid[x, y, 0, :] = outs[n].mean.cpu()
        grid_std[x, y, 0, :] = outs[n].stddev.cpu()

    grid = grid.view(grid.shape[0]*grid.shape[1], *grid.shape[2:])
    grid_std = grid_std.view(grid_std.shape[0]*grid_std.shape[1], *grid_std.shape[2:])

    grid_img = make_grid(grid, nrow=grid_shape)
    grid_std_img = make_grid(grid_std, nrow=grid_shape)
    
    epoch = "" if epoch is None else "_%d"%epoch
    layer = len(model.platent) + layer if layer < 0 else layer
    if not os.path.isdir(out+'/grid'):
        os.makedirs(out+'/grid')
    save_image(grid_img, out+'/grid/grid_%d%s.png'%(layer, epoch))
    save_image(grid_std_img, out+'/grid/std_grid_%d%s.png'%(layer, epoch))


    fig = plt.figure()
    plt.imshow(grid_img.transpose(0,2), aspect="auto")

    return [fig], [fig.axes]


def image_export(dataset, model, label=None, n_rows=None, ids=None, out=None, partition=None, n_points=10, **kwargs):
    if ids is None:
        if partition:
            current_partition = dataset.partitions[partition]
            ids = current_partition[np.random.permutation(len(current_partition))[:n_points]]
        else:
            ids = np.random.permutation(dataset.data.shape[0])[:n_points]

    # get item ids
    if not ids is None:
        n_rows = int(np.sqrt(len(ids)))
        reconstruction_ids = ids
    else:
        n_rows = n_rows or 5
        reconstruction_ids = np.random.permutation(dataset.data.shape[0])[:n_rows**2]
    
    # get corresponding images
    images = dataset.data[reconstruction_ids]
    if not label is None:
        if not issubclass(type(label), list):
            label = [label]
        metadata = {t: dataset.metadata[t][reconstruction_ids] for t in label} 
    else:
        metadata = None
        
    # forward
    out_image = model.forward(images, y=metadata)['x_params'].mean
    out_image = out_image.reshape(out_image.size(0), 1, 28, 28)
    out_grid = make_grid(out_image, nrow=n_rows)
    if out:
        save_image(out_grid, out+'_grid.png')
        
    fig = plt.figure()
    plt.imshow(np.transpose(out_grid.cpu().detach().numpy(), (1,2,0)), aspect='auto')
    del out_image
    return [fig], [fig.axes]


def plot_tf_reconstructions(dataset, model, label=None, n_points=10, out=None, preprocess=True, preprocessing=None, partition=None,
                         name=None, loader=None, ids=None, plot_multihead=False, reinforcers=None, **kwargs):

    if partition is not None:
        dataset = [dataset[0].retrieve(partition), dataset[1].retrieve(partition)]

    if ids is None:
        if issubclass(type(dataset[0].data), list):
            full_id_list = np.array(range(len(dataset[0])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids
        else:
            full_id_list = np.array(range(len(dataset[0])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids

    # get audio data
    Loader = loader if loader else DataLoader
    loader_audio = Loader(dataset[0], None, ids=ids, is_sequence=model[0].take_sequences, tasks=label)
    loader_symbol = Loader(dataset[1], None, ids=ids, is_sequence=model[1].take_sequences, tasks=label)
    data_audio, metadata_audio = next(loader_audio.__iter__())
    #TODO why
    data_symbol = [np.array(x)[ids] for x in dataset[1].data]
    if preprocess:
        if preprocessing[0]:
            data_audio_pp = preprocessing[0](data_audio)
        if preprocessing[1]:
            data_symbol_pp = preprocessing[1](data_symbol)

    else:
        data_audio_pp = data_audio
        data_symbol_pp = data_symbol


    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    
    with torch.no_grad():
        audio_out = model[0].forward(data_audio_pp, y=metadata_audio, **add_args)
        symbol_out = model[1].forward(data_symbol_pp, y=metadata_audio, **add_args)
        z_audio_tf = symbol_out['z_params_enc'][-1].mean
        if model[0].take_sequences:
            z_audio_tf = z_audio_tf.unsqueeze(1)
        z_audio_tf = audio_out['z_enc'][:-1] + [z_audio_tf]
        audio_tf = model[0].decode(z_audio_tf)
        symbol_tf = model[1].decode(audio_out['z_params_enc'][-1].mean)
        #audio_tf = model[0].decode(symbol_out['z_enc'][0])
        #symbol_tf = model[1].decode(audio_out['z_enc'][0])

    if reinforcers is not None:
        if reinforcers[0] is not None:
            audio_out = reinforcers[0].forward(audio_out)

    # compute transfers
    # WARNING here the conv module should work without *[0]

    n_examples = n_points
    n_symbols = len(model[1].pinput)
    grid = plt.GridSpec(n_examples*2, 3 * len(model[1].pinput))
    fig = plt.figure(figsize=(14,8))

    data_audio = data_audio_pp.cpu().detach().numpy()
    audio_out = audio_out['x_params'].mean.cpu().detach().numpy()
    audio_tf_out =audio_tf[0]['out_params'].mean.cpu().detach().numpy()
    is_image = False

    if preprocessing[0] is not None:
        data_audio = preprocessing[0].invert(data_audio)
        audio_out = preprocessing[0].invert(audio_out)
        audio_tf_out = preprocessing[0].invert(audio_tf_out)

    if len(audio_out.shape) > 2:
        if (audio_out.shape[1] == 1 and not model[0].take_sequences): 
            audio_out = np.squeeze(audio_out)
            audio_tf_out = np.squeeze(audio_tf_out)
        elif  (audio_out.shape[2]==1 and model[0].take_sequences):
            audio_out = np.squeeze(audio_out)
            audio_tf_out = np.squeeze(audio_tf_out)
            is_image = True
        else:
            is_image = True
    # data_symbol = [d.squeeze().cpu().detach().numpy() for d in data_symbol]

    for i in range(n_examples):
        # plot original signal
        ax1 = fig.add_subplot(grid[2*i, :n_symbols])
        if not is_image:
            ax1.plot(data_audio[i])
        else:
            ax1.imshow(data_audio[i], aspect='auto')
        # plot original symbols
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i+1, l])
            current_ax.plot(data_symbol[l][i])

        # reconstructed signals
        ax2 = fig.add_subplot(grid[2*i, n_symbols:2*n_symbols])
        if is_image:
            ax2.imshow(audio_out[i], aspect='auto')
        else:
            ax2.plot(data_audio[i], linewidth=0.5)
            ax2.plot(audio_out[i])
        # reconstructed labels
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i+1, n_symbols+l])
            current_ax.plot(data_symbol[l][i], linewidth=0.5)
            current_ax.plot(symbol_out['x_params'][l].probs[i].cpu().detach().numpy())
        # transferred data
        ax3 = fig.add_subplot(grid[2*i+1, 2*n_symbols:])
        if is_image:
            ax3.imshow(audio_tf_out[i], aspect='auto')
        else:
            ax3.plot(data_audio[i], linewidth=0.5)
            ax3.plot(audio_tf_out[i])
        for l in range(n_symbols):
            current_ax = fig.add_subplot(grid[2*i, 2*n_symbols+l])
            current_ax.plot(data_symbol[l][i], linewidth=0.5)
            current_ax.plot(symbol_tf[0]['out_params'][l].probs[i].cpu().detach().numpy())

    if out:
        fig.savefig(out+'.pdf', format='pdf')

    del audio_out; del symbol_out
    gc.collect(); gc.collect();

    return [fig], fig.axes


def plot_mx_reconstructions(datasets, model, solo_models=None, n_examples=3, out=None, random_mode='uniform'):
    ids = []; datas = []
    for d in datasets:
        ids.append( np.random.permutation(d.data.shape[0])[:n_examples] )
        datas.append(d.data[ids[-1]])
    if random_mode == 'uniform':
        random_weights = np.random.random((len(datasets), n_examples))
    elif random_mode == 'constant':
        random_weights = np.ones((len(datasets), n_examples))
    datas = np.array(datas)
    mixtures = np.sum( np.expand_dims(random_weights, -1) * datas, 0)

    vae_out = model.forward(mixtures)

    cmap = core.get_cmap(n_examples)
    fig, ax = plt.subplots(n_examples+1, len(datasets)+1, figsize=(20,10))
    if ax.ndim == 1:
        ax = np.array([ax])

    for i in range(n_examples):
        for j in range(len(datasets)):
            ax[i, j].plot(datas[j, i])
            ax[i, j].plot(vae_out['x_solo_params'][j].mean[i].detach().cpu().numpy(), linewidth=0.5)
            ax[i, j].set_title('weight : %f'%random_weights[j,i])

    for j in range(len(datasets)):
        z_projs = vae_out['z_params_enc'][0][j].mean.cpu().detach().numpy()
        ax[n_examples, j].scatter(z_projs[:, 0], z_projs[:, 1], c=cmap(np.arange(n_examples)))
        """
        if solo_models:
            z_projs_solo = solo_outs[j]['z_params_enc'][0][0].cpu().detach().numpy()
            ax[n_examples, j].scatter(z_projs_solo[:, 0], z_projs_solo[:, 1], marker = 'v', c=cmap(np.arange(n_examples)))
        """

    for i in range(n_examples):
        ax[i, len(datasets)].plot(mixtures[i])
        ax[i, len(datasets)].plot(vae_out['x_params'][i].detach().cpu().numpy())
        ax[i, len(datasets)].set_title('inferred weigths : %s'%vae_out['mixture_coeff'].mean[i, :])
#        if solo_models:
    if out:
        fig.savefig(out+'.pdf', format='pdf')

    del vae_out
    gc.collect(); gc.collect()
    return fig, ax


def plot_mx_latent_space(datasets, model, n_points=None, out=None, tasks=None):
    ids = []; datas = []
    for d in datasets:
        n_examples = n_points
        if n_examples is None:
            n_examples = d.data.shape[0]
        ids.append( np.random.permutation(d.data.shape[0])[:n_examples] )
        datas.append(d.data[ids[-1]])

    min_size = min([i.shape[0] for i in ids])
    ids = [i[:min_size] for i in ids]

    random_weights = np.random.random((len(datasets), min_size))
    datas = np.array(datas)
    mixtures = np.sum(np.expand_dims(random_weights, -1) * datas, 0)

    vae_out = model.forward(mixtures)

    z_out = vae_out['z_params_enc'][0]
    tasks = [None] if tasks is None else tasks

    for t in tasks:
        figs = []; axes = []
        fig, ax = plt.subplots(len(z_out), figsize=(10,10))
        if not issubclass(type(ax), np.ndarray):
            ax = (ax,)
        for i, z in enumerate(z_out):
            handles = []
            if not t is None:
                y = datasets[i].metadata[t].astype('int')[ids[i]]
                cmap = core.get_cmap(len(set(y)))
                colors = cmap(y)
                classes = datasets[i].classes.get(t)
                if not classes is None:
                    reverse_classes = {v:k for k,v in classes.items()}
                    for u in set(y):
                        patch = mpatches.Patch(color=cmap(u), label=reverse_classes[u])
                        handles.append(patch)

            else:
                colors = None
            z_tmp = z.mean.detach().cpu().numpy()
            s_tmp = np.max(z.variance.detach().cpu().numpy(), 1)
            ax[i].scatter(z_tmp[:, 0], z_tmp[:, 1], c=colors, s=15.0*s_tmp)

            if not len(handles) == 0:
                ax[i].legend(handles=handles)

        if not out is None:
            fig.savefig(out+'_%s.pdf'%t, format='pdf')
        figs.append(fig); axes.append(ax)

    return fig, ax


def get_spectral_transform(x, transform='fft', window_size=4096, mel_filters=256):
    if len(x.shape) == 2:
        x = x.unsqueeze(1)
    if transform in ('stft', 'stft-mel'):
        x_fft = torch.stft(x.squeeze(), window=torch.hann_window(window_size, device=x.device), center=True, pad_mode='constant')
        x_fft_real = x_fft[:,:,:, 0]; x_fft_imag = x_fft[:,:,:, 1];
    elif transform in ('fft', 'fft-mel'):
        x_fft = torch.fft(torch.cat([x.squeeze().unsqueeze(-1), torch.zeros_like(x.squeeze().unsqueeze(-1))], dim=-1), 2)
        x_fft_real = x_fft.select(-1, 0); x_fft_imag = x_fft.select(-1, 1);
        x_fft_real = x_fft_real[:, :int(x_fft_real.shape[1]/2+1)];
        x_fft_imag = x_fft_imag[:, :int(x_fft_imag.shape[1]/2+1)];
        window_size = x_fft_real.shape[1]*2

    x_radius = torch.sqrt(x_fft_real.pow(2) + x_fft_imag.pow(2))
    x_angle = torch.atan2(x_fft_real, x_fft_imag+eps)
    if transform in ("stft-mel", 'fft-mel'):
        mel_w = librosa.filters.mel(22050, window_size-1, n_mels = min(mel_filters, window_size))
        mel_weights = torch.from_numpy(mel_w).float().to(x_fft).detach()
        x_radius = torch.bmm(mel_weights.unsqueeze(0).repeat(x_radius.shape[0],1,1), x_radius.unsqueeze(-1)).transpose(1,2)
    return x_radius, x_angle, x_fft_real, x_fft_imag

def plot_spectrograms(dataset, model, label=None, n_points=10, out=None, preprocessing=None, partition=None, ids=None,
                      transform="fft", window_size=2048, mel_filters=256, sample=False, plot_multihead=False):
    # get plotting ids

    if not torch.backends.mkl.is_available():
        print('Error in plot spectrograms : MKL backend not available.')
        return

    n_rows, n_columns = core.get_divs(n_points)
    if ids is None:
        if issubclass(type(dataset.data), list):
            full_id_list = np.array(range(dataset.data[0].shape[0])) if partition is None else np.array(range(len(dataset.partitions[partition])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids
        else:
            full_id_list = np.array(range(dataset.data.shape[0])) if partition is None else np.array(range(len(dataset.partitions[partition])))
            ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids

    # get data
    loader = DataLoader(dataset, None, ids=ids, is_sequence=model.take_sequences)
    data, _ = next(loader.__iter__())
    if issubclass(type(model.pinput), list):
        # if not issubclass(type(dataset.data), list):
        #     data = [dataset.data]
        # data = [d[ids] for d in data]
        preprocessing = preprocessing if not preprocessing is None else [None]*len(dataset.data)
        if not issubclass(type(preprocessing), list):
            preprocessing = [preprocessing]
        for i, pp in enumerate(preprocessing):
            if not pp is None:
                data[i] = preprocessing(data[i])
        else:
            data = [dataset.data[ids]]
            preprocessing = [preprocessing]
    else:
        if preprocessing is not None:
            data = preprocessing(data)

    # in case, get metadata
    metadata = None
    if not label is None:
        if not issubclass(type(label), list):
            label = [label]
        if not label is None:
            metadata = {l: dataset.metadata.get(l)[ids] for l in label}
    else:
        metadata = None

    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    with torch.no_grad():
        vae_out = model.forward(data, y=metadata, **add_args)['x_params']


    if not issubclass(type(vae_out), list):
        vae_out = [vae_out]
    if not issubclass(type(data), list):
        data = [data]

    # plot
    figs = []; axes = []
    nrows, ncolumns = core.get_divs(n_points)
    for i in range(len(vae_out)):
        # multihead_outputs = None
        # if plot_multihead and hasattr(model.decoders[0].out_modules[0], 'current_outs'):
        #     multihead_outputs = model.decoders[0].out_modules[0].current_outs
        if sample:
            current_out = vae_out[i].sample()
        else:
            current_out = vae_out[i].mean
        if preprocessing:
            current_out = preprocessing.invert(current_out)

        spec_orig = get_spectral_transform(data[i], window_size=window_size, mel_filters=mel_filters)
        spec_rec = get_spectral_transform(current_out)
        fig, ax = plt.subplots(nrows, ncolumns)
        if transform in ('fft', 'fft-mel'):
            for j in range(nrows):
                for k in range(ncolumns):
                    ax[j, k].plot(spec_orig[j*ncolumns+k])
                    ax[j, k].plot(spec_rec[j*ncolumns+k])

        if not out is None:
            fig.suptitle('output %d'%i)
            name = out+'.pdf' if len(vae_out) == 1 else out+'_%d.pdf'%i
            fig.savefig(name, format="pdf")
        figs.append(fig); axes.append(ax)

    del data; del vae_out
    return figs, axes

        
################################################
########        LATENT PLOTS
####

def plot_latent2(dataset, model, transformation, n_points=None, tasks=None, classes=None, label=None, balanced=False, legend=True,
                   sample = False, layers=0, color_map="plasma", zoom=10, out=None, verbose=False, preprocessing=None, *args, **kwargs):
    # select points
    if balanced and tasks!=None:
        ids = set()
        
        task_ids = []
#        pdb.set_trace()
        for t in tasks:
            task_ids.append(core.get_balanced_ids(dataset.metadata[t], n_points))
            ids = ids.union(set(task_ids[-1]))
        ids = list(ids)
        task_ids = [np.array([ids.index(x) for x in task_id]) for task_id in task_ids]
    else:
        ids = dataset.data.shape[0] if n_points is None else np.random.permutation(dataset.data.shape[0])[:n_points]
        task_ids = [] if tasks is None else [range(len(ids))]*len(tasks)
        
    ids = np.array(ids)
    
    x = dataset.data[ids]
    if not preprocessing is None:
        x = preprocessing(x)

    data = model.format_input_data(x)
    metadata = model.format_label_data(dataset.metadata[label][ids]) if not label is None else None
    output = model.forward(data, y=metadata, *args, **kwargs)
    figs = []
    
    for layer in layers:
        # make manifold
        if tasks is None:
            fig = plt.figure('latent plot of layer %d'%layer)
            
            current_z = output['z_params_enc'][layer].mean.detach().numpy()
            if current_z.shape(1) > 2:
                current_z = transformation.fit_transform(current_z)
                
            plt.scatter(output['z_params_enc'][layer][:, 0], output['z_params_enc'][layer][:,1])
            plt.title('latent plot of layer %d'%layer)
            if not out is None:
                fig.savefig(out+'_layer%d.svg'%layer, format="svg")
            figs.append(fig)
        else:
            if not issubclass(type(tasks), list):
                tasks = [tasks]
            current_z = output['z_params_enc'][layer].mean.detach().numpy()
            if current_z.shape[1] > 2:
                current_z = transformation.fit_transform(current_z)
            for id_task, task in enumerate(tasks):
                print('-- plotting task %s'%task)
                fig = plt.figure('latent plot of layer %d, task %s'%(layer, task))
    #            pdb.set_trace()
                meta = np.array(dataset.metadata[task])[ids[task_ids[id_task]]]
                _, classes = core.get_class_ids(dataset, task, ids=ids[task_ids[id_task]])
    #            pdb.set_trace()
                
                cmap = core.get_cmap(len(classes))
                current_z = current_z[task_ids[id_task], :] if balanced else current_z
                plt.scatter(current_z[:, 0], current_z[:,1], c=cmap(meta))
                if legend:
                    handles = []
                    if dataset.classes.get(task)!=None:
                        class_names = {v: k for k, v in dataset.classes[task].items()}
                        for cl in classes:
                            patch = mpatches.Patch(color=cmap(cl), label=class_names[cl])
                            handles.append(patch)
                        fig.legend(handles=handles)
                        
                figs.append(fig)
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix
                if not out is None:
                    title = out+'_layer%d_%s.pdf'%(layer, task)
                    fig.savefig(title, format="pdf")
    return figs






def plot_latent3(dataset, model, transformation=None, n_points=None, preprocessing=None, label=None, tasks=None, ids=None, balanced=False, batch_size=None, partition=None, epoch=None,
                   preprocess=True, loader=None, sample = False, layers=None, color_map="plasma", zoom=10, out=None, name=None, legend=True, centroids=False, sequence=False, prediction=None, *args, **kwargs):
    '''
    3-D plots the latent space of a model
    :param dataset: `vschaos.data.Dataset` object containing the data
    :param model: `vschaos.vaes.AbstractVAE` child
    :param transformation: `vschaos.visualize_dimred.Embedding` object used to project on 3 dimensions (if needed)
    :param n_points: number of points plotted
    :param preprocessing: preprocessing used before encoding
    :param label: conditioning data
    :param tasks: tasks plotted (None for no task-related coloring)
    :param ids: plot given ids from dataset (overrides n_points and balanced options)
    :param classes:
    :param balanced: balance data for each task
    :param batch_size: batch size of the loader
    :param preprocess:
    :param loader: class of the data loader used
    :param sample: sample from distribution (if `False`: takes the mean)
    :param layers: latent layers to plot (default : all)
    :param color_map: color map used for coloring (default: plasma)
    :param zoom: weight of each point radius (default: 10)
    :param out: if given, save plots at corresponding places
    :param legend: plots legend if `True`
    :param centroids: plots centroids for each class
    :return: list(figure), list(axis)
    '''

    ### prepare data IDs
    tasks = checklist(tasks)
    if len(tasks) == 0 or tasks is None:
        tasks = [None] 
    full_ids = CollapsedIds()
    if partition:
        dataset = dataset.retrieve(partition)
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])
    if tasks == [None]:
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
        nclasses = {None:None}; classes_ids = {}
    else:
        #if n_points is not None:
        #    ids = np.random.permutation(len(dataset.data))[:n_points]
        class_ids = {}; nclasses = {}
        for t in tasks:
            class_ids[t], nclasses[t] = core.get_class_ids(dataset, t, balanced=balanced, ids=ids, split=True)
            full_ids.add(t,np.concatenate(list(class_ids[t].values())))

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks = label)
    # forward!
    output = []
    with torch.no_grad():
        for x,y in loader:
            if not preprocessing is None and preprocess:
                x = preprocessing(x)
            output.append(decudify(model.encode(model.format_input_data(x), y=y, return_shifts=False, *args, **kwargs)))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)

    ### plot!
    figs = {}; axes = {}
    layers = layers or range(len(model.platent))
    if out:
        out += "/latent"
        check_dir(out)
    # iterate over layers
    for layer in layers:
        # sample = True -> sample distirbution; sample = False -> take mean of distribution
        vae_out[layer]['out'] = checklist(vae_out[layer]['out'])
        vae_out[layer]['out_params'] = checklist(vae_out[layer]['out_params'])
        if sample:
            full_z = np.concatenate([x.cpu().detach().numpy() for x in vae_out[layer]['out']], axis=-1)
            full_var = np.ones_like(full_z) * 1e-3
        else:
            full_z = np.concatenate([x.mean.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)
            full_var = np.concatenate([x.variance.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)
            full_var = np.mean(full_var, tuple(range(1,full_var.ndim)))*zoom
        # transform in case
        if full_z.shape[-1] > 3 and not sequence:
            assert transformation, 'if dimensionality > 3 please specify the transformation keyword'
            if issubclass(type(transformation), list):
                transformation = transformation[layer]
            if len(full_z.shape) == 3:
                full_z = full_z.reshape(full_z.shape[0]*full_z.shape[1], *full_z.shape[2:])
            if issubclass(type(transformation), type):
                full_z = transformation(n_components=3).fit_transform(full_z)
            else:
                full_z = transformation.transform(full_z)
           
        # iteration over tasks
        for task in tasks:
            print('-- plotting task %s'%task)
            if task:
                meta = np.array(dataset.metadata[task])[full_ids.ids[task]]
            else:
                meta = None; class_ids = {None:None}; classes=None; class_names=None

            legend = legend and len(class_ids)<12
            if hasattr(meta[0], '__iter__'):
                for k, v in class_ids[task].items():
                    class_names = {v:k for k, v in dataset.classes[task].items()}
                    current_ids = full_ids.transform(v)
                    ghost_ids = np.array(list(filter(lambda x: x not in current_ids, range(full_z.shape[0]))))
                    fig, ax = core.plot(full_z[current_ids], meta=[k]*len(v), var=full_var[full_ids.transform(v)], classes=nclasses[task], class_ids={k:v}, class_names=class_names[k], centroids=centroids,legend=False, sequence=sequence, shadow_z =full_z[ghost_ids])
                    # register and export
                    fig_name = 'layer %d / task %s / class %s'%(layer, task, class_names[k]) if task else 'layer%d'%layer
                    fig.suptitle(fig_name)
                    name = name or 'latent'
                    title = '%s_layer%d_%s_%s'%(name,layer, task, class_names[k])
                    figs[title] = fig; axes[title] = ax
                    if not out is None:
                        title = '%s%s.pdf'%(out, title)
                        fig.savefig(title, format="pdf")
            else:
                class_names = {} if len(tasks) == 0 else {v:k for k, v in dataset.classes[task].items()}
                fig, ax = core.plot(full_z[full_ids.get_ids(task)], meta=meta, var=full_var[full_ids.get_ids(task)], classes=nclasses[task], class_ids=class_ids, class_names=class_names, centroids=centroids, legend=legend)
                # register and export
                fig_name = 'layer %d / task %s'%(layer, task) if task else 'layer%d'%layer
                fig.suptitle(fig_name)
                name = name or 'latent'
                title = '%slayer%d_%s'%(name, layer, task)
                figs[title] = fig; axes[title] = ax
                print(figs.keys())
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix
                if not out is None:
                    fig.savefig(f"{out}/{title}{suffix}.pdf", format="pdf")
    gc.collect(); gc.collect()
    return figs, axes


def plot_latent_dim(dataset, model, label=None, tasks=None, n_points=None, layers=None, legend=True, out=None, ids=None, transformation=None, name=None,
                    partition=None, epoch=None, preprocess=True, loader=None, batch_size=None, balanced=True, preprocessing=None, sample=False, *args, **kwargs):
    ### prepare data IDs
    tasks = checklist(tasks)
    if len(tasks) == 0 or tasks is None:
        tasks = [None]
    full_ids = CollapsedIds()
    if partition:
        dataset = dataset.retrieve(partition)
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])

    if tasks == [None]:
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
        class_ids = {None:None}; nclasses = {None:[]}
    else:
        # if n_points is not None:
        #    ids = np.random.permutation(len(dataset.data))[:n_points]
        class_ids = {};
        nclasses = {}
        for t in tasks:
            class_ids[t], nclasses[t] = core.get_class_ids(dataset, t, balanced=balanced, ids=ids, split=True)
            full_ids.add(t, np.concatenate(list(class_ids[t].values())))

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks=label)
    # forward!
    output = []
    with torch.no_grad():
        for x, y in loader:
            if not preprocessing is None and preprocess:
                x = preprocessing(x)
            output.append(
                decudify(model.encode(model.format_input_data(x), y=y, return_shifts=False, *args, **kwargs)))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)

    ### plot!
    figs = {}; axes = {}
    layers = layers or list(range(len(model.platent)))
    transformation = checklist(transformation)
    if out:
        out += "/dims"
        check_dir(out)
    # iterate over layers
    for layer in layers:
        # sample = True -> sample distirbution; sample = False -> take mean of distribution
        vae_out[layer]['out'] = checklist(vae_out[layer]['out'])
        vae_out[layer]['out_params'] = checklist(vae_out[layer]['out_params'])
        if sample:
            full_z = np.concatenate([x.cpu().detach().numpy() for x in vae_out[layer]['out']], axis=-1)
        else:
            full_z = np.concatenate([x.mean.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)
        full_var = np.concatenate([None if not hasattr(x, "stddev") else x.stddev for x in vae_out[layer]['out_params']], axis=0)
        # transform in case
        for reduction in transformation:
            full_z_t = full_z
            if reduction:
                if full_z.shape[-1] > 3:
                    if issubclass(type(reduction), list):
                        reduction = reduction[layer]
                    if issubclass(type(reduction), type):
                        full_z_t = reduction(n_components=3).fit_transform(full_z)
                    else:
                        full_z_t = reduction.transform(full_z)

            # iteration over tasks
            for task in tasks:
                if task:
                    meta = np.array(dataset.metadata[task])[full_ids.ids[task]]
                    class_names = {v: k for k, v in dataset.classes[task].items()}
                else:
                    meta = {}
                    class_ids = {None: None}; class_names = None
                fig, ax = core.plot_dims(full_z_t[full_ids.get_ids(task)], meta=meta, classes=nclasses[task],
                                         class_ids=class_ids, class_names=class_names, legend=legend, var=full_var[full_ids.get_ids(task)])

                # register and export
                fig_name = 'layer %d / task %s' % (layer, task) if task else 'layer%d' % layer
                fig.suptitle(fig_name)

                name = name or 'dims'
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix

                title = '%s_layer%d_%s_%s' % (name, layer, str(reduction), task)
                if not out is None:
                    fig.savefig(f'{out}/{title}{suffix}.pdf', format="pdf")
                figs[title] = fig; axes[title] = ax

    return figs, axes

def plot_latent_consistency(dataset, model, label=None, tasks=None, n_points=None, layers=None, legend=True, out=None, ids=None, transformation=None, name=None,
                     epoch=None, preprocess=True, loader=None, batch_size=None, partition=None, preprocessing=None, sample=False, *args, **kwargs):

    assert len(model.platent) > 1, "plot_latent_consistency is only made for hierarchical models"
    # get plotting ids
    if partition is not None:
        dataset = dataset.retrieve(partition)
    n_rows, n_columns = core.get_divs(n_points)
    if ids is None:
        full_id_list = np.arange(len(dataset))
        ids = full_id_list[np.random.permutation(len(full_id_list))[:n_points]] if ids is None else ids

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=ids, tasks=label)
    # forward!
    output = []
    with torch.no_grad():
        for x, y in loader:
            if not preprocessing is None and preprocess:
                x = preprocessing(x)
            out_tmp = model.forward(x, y=y, **kwargs)
            output.append(decudify(out_tmp))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)

    ### plot!
    figs = {}; axes = {}
    layers = layers or list(range(len(model.platent)))
    transformation = checklist(transformation)
    if out:
        out += "/consistency"
        check_dir(out)
    # iterate over layers
    for layer in layers:
        # sample = True -> sample distirbution; sample = False -> take mean of distribution
        # transform in case
        for reduction in transformation:
            if layer >= len(vae_out['z_params_dec']):
                continue
            z_enc_params = vae_out['z_params_enc'][layer]; z_dec_params = vae_out['z_params_dec'][layer]
            full_z_enc = z_enc_params.mean; full_z_dec = z_dec_params.mean
            full_z_var_enc = None if not hasattr(z_enc_params, "stddev") else z_enc_params.stddev.detach().cpu().numpy()
            full_z_var_dec = None if not hasattr(z_dec_params, "stddev") else z_dec_params.stddev.detach().cpu().numpy()
            if reduction:
                if issubclass(type(reduction), list):
                    reduction = reduction[layer]
                if issubclass(type(reduction), type):
                    reduction = reduction(n_components=3).fit(np.concatenate([full_z_enc, full_z_dec], axis=0))
                    full_z_enc = reduction(n_components=3).transform(full_z_enc)
                    full_z_dec = reduction(n_components=3).transform(full_z_dec)
                else:
                    full_z_enc = reduction.transform(full_z_enc)
                    full_z_dec = reduction.transform(full_z_dec)

            full_z_enc = full_z_enc.detach().cpu().numpy(); full_z_dec = full_z_dec.detach().cpu().numpy()
            # iteration over tasks
            fig, ax = core.plot_pairwise_trajs([full_z_enc, full_z_dec], var=[full_z_var_enc, full_z_var_dec])

            # register and export
            for i in range(len(ids)):
                fig_name = dataset.files[ids[i]] or "consist_%d"%i
                fig[i].suptitle(fig_name)
                name = name or 'dims'
                title = '/%s_%s_%s_%s'%(name, str(reduction), layer, i)
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix
                if not out is None:
                    fig[i].savefig(f"{out}/{title}{suffix}.pdf", format="pdf")
                figs[title] = fig[i]; axes[title] = ax[i]

    return figs, axes



def plot_latent_stats(dataset, model, label=None, tasks=None, n_points=None, layers=None, legend=True, out=None, preprocess=True,
                      loader=None, epoch=None, partition=None, batch_size=None, balanced=True, preprocessing=None, *args, **kwargs):

    ### prepare data IDs
    ids = None # points ids in database
    if partition:
        dataset = dataset.retrieve(partition)
    layers = layers or range(len(model.platent))
    tasks = checklist(tasks)
    if len(tasks) == 0:
        tasks = None
    full_ids = CollapsedIds()
    if tasks is None:
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
    else:
        class_ids = {}; class_names = {}
        for t in tasks:
            class_ids[t], class_names[t]= core.get_class_ids(dataset, t, balanced=balanced, ids=ids, split=True)
            full_ids.add(t,np.concatenate(list(class_ids[t].values())))

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks = label)
    # forward!
    output = []
    with torch.no_grad():
        for x,y in loader:
            if not preprocessing is None and preprocess:
                x = preprocessing(x)
            output.append(decudify(model.encode(model.format_input_data(x), y=y, return_shifts=False, *args, **kwargs)))
    torch.cuda.empty_cache()
    vae_out = merge_dicts(output)

    figs = {}; axes = {}
    if out:
        out += "/stats/"
        check_dir(out)
    for layer in layers:
        latent_dim = vae_out[layer]['out'].shape[-1]
        id_range = np.array(list(range(latent_dim)))
        if tasks is None:
            fig = plt.figure('latent statistics for layer %d'%layer)
            ax1 = fig.add_subplot(211); ax1.set_title('variance of latent positions')
            ax2 = fig.add_subplot(212); ax2.set_title('mean of variances per axis')
            pos_var = [np.std(vae_out[layer]['out_params'].mean.cpu().detach().numpy(), 0)]
            var_mean = [np.mean(vae_out[layer]['out_params'].variance.cpu().detach().numpy(), 0)]
            width = 1/len(pos_var)
            cmap = core.get_cmap(len(pos_var))
            for i in range(len(pos_var)):
                ax1.bar(id_range+i*width, pos_var[i], width)
                ax2.bar(id_range+i*width, var_mean[i], width)
    #        ax1.set_xticklabels(np.arange(latent_dim), np.arange(latent_dim))
    #        ax2.set_xticklabels(np.arange(latent_dim), np.arange(latent_dim))

            title = "stats_layer_%d"%layer
            if not out is None:
                fig.savefig(out+'_layer%d.pdf'%layer, format="pdf")
            figs[title] = fig; axes[title] = fig.axes
        else:
            if not issubclass(type(tasks), list):
                tasks = [tasks]
            for t, task in enumerate(tasks):
                print('-- plotting task %s'%task)
                fig = plt.figure('latent statistics for layer %d, task %s'%(layer, task))
                ax1 = fig.add_subplot(211); ax1.set_title('variance of latent positions')
                ax2 = fig.add_subplot(212); ax2.set_title('mean of variances per axis')
                # get data
                pos_var = []; var_mean= [];
                width = 1/len(class_ids[task].keys())
                cmap = core.get_cmap(len(class_ids[task].keys()))
                zs = vae_out[layer]['out_params']
                handles = []; counter=0
                class_names = {v:k for k,v in dataset.classes[task].items()}
                for i, c in class_ids[task].items():
                    reg_axis = tuple(np.arange(len(vae_out[layer]['out'].shape)-1))
                    pos_var.append(np.std(zs.mean[full_ids.transform(c)].cpu().detach().numpy(), reg_axis))
                    var_mean.append(np.mean(zs.variance[full_ids.transform(c)].cpu().detach().numpy(), reg_axis))
                    ax1.bar(id_range+counter*width, pos_var[-1], width, color=cmap(counter))
                    ax2.bar(id_range+counter*width, var_mean[-1], width, color=cmap(counter))
                    if legend:
                        patch = mpatches.Patch(color=cmap(counter), label=class_names[i])
                        handles.append(patch)
                    counter += 1
                if legend:
                    fig.legend(handles=handles)

                title = 'stats_layer%d_%s.pdf'%(layer, task)
                figs[title] = fig; axes[title] = fig.axes
                suffix = ""
                suffix = "_"+str(partition) if partition is not None else suffix
                suffix = suffix+"_%d"%epoch if epoch is not None else suffix
                if not out is None:
                    fig.savefig(f"{out}/{title}{suffix}.pdf", format="pdf")

    # plot histograms
    return figs, axes
        

def plot_latent_dists(dataset, model, label=None, tasks=None, bins=20, layers=[0], n_points=None, dims=None, legend=True, split=False, out=None, relief=True, ids=None, **kwargs):
    # get data ids
    if n_points is None:
        ids = np.arange(dataset.data.shape[0]) if ids is None else ids
        data = dataset.data
        y = dataset.metadata.get(label)
    else:
        ids = np.random.permutation(dataset.data.shape[0])[:n_points] if ids is None else ids
        data = dataset.data[ids]
        y = dataset.metadata.get(label)
        if not y is None:
            y = y[ids]
    y = model.format_label_data(y)
    data = model.format_input_data(data);
    
    if dims is None:
        dims = list(range(model.platent[layer]['dim']))
        
    # get latent space
    with torch.no_grad():
        vae_out = model.encode(data, y=y)
        # get latent means of corresponding parameters
    
    # get  
    figs = []
    
    for layer in layers:
        zs = model.platent[layer]['dist'](*vae_out[0][layer]).mean.cpu().detach().numpy()
        if split:
            if tasks is None:
                for dim in dims:
                    fig = plt.figure('dim %d'%dim, figsize=(20,10))
                    hist, edges = np.histogram(zs[:, dim], bins=bins)
                    plt.bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                    if not out is None:
                        prefix = out.split('/')[:-1]
                        fig.savefig(prefix+'/dists/'+out.split('/')[-1]+'_%d_dim%d.svg'%(layer, dim))
                    figs.append(fig)
            else:
                if not os.path.isdir(out+'/dists'):
                    os.makedirs(out+'/dists')
                for t in range(len(tasks)):
                    class_ids, classes = get_class_ids(dataset, tasks[t], ids=ids)
                    cmap = get_cmap(len(class_ids))
                    for dim in dims:
                        fig = plt.figure('dim %d'%dim, figsize=(20, 10))
                        ax = fig.gca(projection='3d') if relief else fig.gca()
                        for k, cl in enumerate(class_ids):
                            hist, edges = np.histogram(zs[cl, dim], bins=bins)
                            colors = cmap(k)
                            if relief:
                                ax.bar3d(edges[:-1], k*np.ones_like(hist), np.zeros_like(hist), edges[1:]-edges[:-1], np.ones_like(hist), hist, color=colors)
                                ax.view_init(30,30)
                            else:
                                ax.bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                        if legend and not dataset.classes.get(tasks[t]) is None:
                            handles = []
                            class_names = {v: k for k, v in dataset.classes[tasks[t]].items()}
                            for i in classes:
                                patch = mpatches.Patch(color=cmap(i), label=class_names[i])
                                handles.append(patch)
                            fig.legend(handles=handles)
                        if not out is None:
                            prefix = out.split('/')[:-1]
                            fig.savefig('/'.join(prefix)+'/dists/'+out.split('/')[-1]+'_%d_%s_dim%d.svg'%(layer,tasks[t], dim))
    #                    plt.close('all')
                        figs.append(fig)
        else:
            if tasks is None:
                dim1, dim2 = get_divs(len(dims))
                fig, axes = plt.subplots(dim1, dim2, figsize=(20,10))
                for i in range(axes.shape[0]):
                    for j in range(axes.shape[1]):
                        current_id = i*dim2 + j
                        hist, edges = np.histogram(zs[:, dims[current_id]], bins=bins)
                        axes[i,j].bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                        axes[i,j].set_title('axis %d'%dims[current_id])
                if not out is None:
                    prefix = out.split('/')[:-1]
                    fig.savefig(out+'_0.svg'%layer)
                figs.append(fig)
            else:
                dim1, dim2 = get_divs(len(dims))
                for t in range(len(tasks)):
                    class_ids, classes = get_class_ids(dataset, tasks[t], ids=ids)
                    cmap = get_cmap(len(class_ids))
                    if relief:
                        fig, axes = plt.subplots(dim1, dim2, figsize=(20,10), subplot_kw={'projection':'3d'})
                    else:
                        fig, axes = plt.subplots(dim1, dim2, figsize=(20,10))
                        
    #                pdb.set_trace()
                    for i in range(axes.shape[0]):
                        dim_y = 0 if len(axes.shape)==1 else axes.shape[1]
                        for j in range(dim_y):
                            current_id = i*dim2 + j
                            for k, cl in enumerate(class_ids):
                                hist, edges = np.histogram(zs[cl, dims[current_id]], bins=bins)
                                colors = cmap(k)
                                if relief:
                                    axes[i,j].bar3d(edges[:-1], k*np.ones_like(hist), np.zeros_like(hist), edges[1:]-edges[:-1], np.ones_like(hist), hist, color=colors, alpha=0.1)
                                    axes[i,j].view_init(30,30)
                                else:
                                    axes[i,j].bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                                axes[i,j].set_title('axis %d'%dims[current_id])
                            
                    if legend and not dataset.classes.get(tasks[t]) is None:
                        handles = []
                        class_names = {v: k for k, v in dataset.classes[tasks[t]].items()}
                        for i in classes:
                            patch = mpatches.Patch(color=cmap(i), label=class_names[i])
                            handles.append(patch)
                        fig.legend(handles=handles)
    
                    if not out is None:
                        prefix = out.split('/')[:-1]
                        fig.savefig(out+'_%d_%s.svg'%(layer, tasks[t]))
                    figs.append(fig)
    return figs



def plot_latent_trajs(dataset, model, n_points=None, preprocessing=None, label=None, tasks=None, balanced=False, batch_size=None,
                   partition=None, epoch=None, preprocess=True, loader=None, sample = False, layers=None, out=None, name=None, legend=True, centroids=False, *args, **kwargs):
    '''
    3-D plots the latent space of a model
    :param dataset: `vschaos.data.Dataset` object containing the data
    :param model: `vschaos.vaes.AbstractVAE` child
    :param transformation: `vschaos.visualize_dimred.Embedding` object used to project on 3 dimensions (if needed)
    :param n_points: number of points plotted
    :param preprocessing: preprocessing used before encoding
    :param label: conditioning data
    :param tasks: tasks plotted (None for no task-related coloring)
    :param ids: plot given ids from dataset (overrides n_points and balanced options)
    :param classes:
    :param balanced: balance data for each task
    :param batch_size: batch size of the loader
    :param preprocess:
    :param loader: class of the data loader used
    :param sample: sample from distribution (if `False`: takes the mean)
    :param layers: latent layers to plot (default : all)
    :param color_map: color map used for coloring (default: plasma)
    :param zoom: weight of each point radius (default: 10)
    :param out: if given, save plots at corresponding places
    :param legend: plots legend if `True`
    :param centroids: plots centroids for each class
    :return: list(figure), list(axis)
    '''

    ### prepare data IDs
    ids = None # points ids in database
    tasks = checklist(tasks)
    if len(tasks) == 0 or tasks is None:
        raise TypeError('taks must not be %s'%tasks)
    full_ids = CollapsedIds()
    if partition:
        dataset = dataset.retrieve(partition)
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])
    if tasks == [None]:
        full_ids.add(None, ids if ids is not None else np.random.permutation(len(dataset.data))[:n_points])
    else:
        #if n_points is not None:
        #    ids = np.random.permutation(len(dataset.data))[:n_points]
        for t in tasks:
            full_ids.add(t, core.get_class_ids(dataset, t, balanced=balanced, ids=ids)[0])

    ### forwarding
    if not issubclass(type(label), list) and not label is None:
        label = [label]
    # preparing dataloader
    Loader = loader or DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks = label)
    # forward!
    output = []
    with torch.no_grad():
        for x,y in loader:
            if not preprocessing is None and preprocess:
                x = preprocessing(x)
            output.append(decudify(model.encode(model.format_input_data(x), y=y, return_shifts=False, *args, **kwargs)))
    vae_out = merge_dicts(output)
    torch.cuda.empty_cache()

    ### plot!
    figs = []; axes = []
    layers = layers or range(len(model.platent))
    if out:
        out += "/latent_trajs"
        check_dir(out)
    # iterate over layers
    for layer in layers:
        # sample = True -> sample distirbution; sample = False -> take mean of distribution
        vae_out[layer]['out'] = checklist(vae_out[layer]['out'])
        vae_out[layer]['out_params'] = checklist(vae_out[layer]['out_params'])
        if sample:
            full_z = np.concatenate([x.cpu().detach().numpy() for x in vae_out[layer]['out']], axis=-1)
            full_var = np.ones_like(full_z) * 1e-3
        else:
            full_z = np.concatenate([x.mean.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)
            full_var = np.concatenate([x.variance.cpu().detach().numpy() for x in vae_out[layer]['out_params']], axis=0)

        # iteration over tasks
        for task in tasks:
            print('-- plotting task %s'%task)
            class_ids, classes = core.get_class_ids(dataset, task, ids=full_ids.ids[task], split=True)
            class_names = {v: k for k, v in dataset.classes[task].items()}

            n_rows, n_columns = core.get_divs(full_z.shape[1])
            fig, axis = plt.subplots(n_rows, n_columns, figsize=(10,10))
            if n_rows == 1:
                axis = axis[np.newaxis, :]
            if n_columns==1:
                axis = axis[:, np.newaxis]
            plt.gca()
            full_z_sorted = [full_z[full_ids.transform(class_ids[i])] for i in classes]
            full_var_sorted =[full_var[full_ids.transform(class_ids[i])] for i in classes]
            for i in range(n_rows):
                for j in range(n_columns):
                    x = np.arange(len(classes))+0.5
                    # draw means
                    axis[i,j].bar(x, np.array([full_var_sorted[c][:, i*n_columns+j] for c in range(len(class_ids))]).mean(1), alpha=0.3, color='r')
                    axis[i,j].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))
                    axis[i,j].yaxis.tick_right()
                    axis[i,j].yaxis.set_ticks_position('right')
                    axis[i,j].yaxis.set_tick_params(labelsize='x-small')
                    if i < n_rows - 1:
                        axis[i,j].xaxis.set_ticks_position('none')
                    else:
                        plt.xticks(x, [class_names[c] for c in classes])
                        axis[i,j].xaxis.set_tick_params(labelsize="x-small", rotation=45)
                    # draw variances
                    axis_b = axis[i,j].twinx()
                    axis_b.plot(x, np.array([full_z_sorted[c][:, i*n_columns+j] for c in range(len(class_ids))]).mean(1), color='r')
                    axis_b.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                    axis_b.yaxis.set_tick_params(labelsize='x-small')
                    axis_b.yaxis.tick_left()
                    if i < n_rows - 1:
                        axis_b.xaxis.set_ticks_position('none')
                    else:
                        plt.xticks(x, [class_names[c] for c in classes])
                        axis_b.xaxis.set_tick_params(labelsize="x-small", rotation=45)

            # register and export
            fig_name = 'layer %d / task %s'%(layer, task)  if task else 'layer%d'%layer
            fig.suptitle(fig_name)

            name = name or 'trajs'
            title = "%s_layer%d_%s"%(name,layer, task)
            figs[title] = fig; axes[title] = axis
            suffix = ""
            suffix = "_"+str(partition) if partition is not None else suffix
            suffix = suffix+"_%d"%epoch if epoch is not None else suffix
            if not out is None:
                title = f"{out}/{title}{suffix}.pdf"
                fig.savefig(title, format="pdf")

    gc.collect(); gc.collect()
    return figs, axes





################################################
########        LOSSES PLOTS
####


def plot_losses(*args, loss=None, out=None, separated=False, axis="time", partition=None, epoch=None, **kwargs):
    assert loss
    assert axis in ['time', 'epochs']
    # get set and loss names
    set_names = loss.loss_history.keys()
    loss_names = list(set(sum([list(f.keys()) for f in loss.loss_history.values()],[])))
    # get number of graphs

    n_rows, n_columns = core.get_divs(len(loss_names))
    figs = {}; axes = {}
    if not separated:
        fig, ax = plt.subplots(n_rows, n_columns, figsize=(15,10))
        figs['losses'] = fig; axes['losses'] = ax
    else:
        fig = [plt.Figure() for i in range(len(loss_names))]
        ax = [f.add_subplot(1,1,1) for f in fig]
        ax = np.array(ax).reshape(n_rows, n_columns)
        for i in range(len(fig)):
            figs[loss_names[i]] = fig[i]; axes[loss_names[i]] = ax[i]

    if n_columns == 1:
        ax = np.expand_dims(ax, 1)
    elif n_rows == 1:
        ax = np.expand_dims(ax, 0)
    for i in range(n_rows):
        for j in range(n_columns):
            current_idx = i*n_columns + j
            current_loss = loss_names[current_idx]
            plots = []
            for k in set_names:
                values = loss.loss_history[k].get(current_loss)['values']
                times = loss.loss_history[k].get(current_loss).get('time')
                x = times if axis == 'time' and times is not None else range(len(values))
                if values is not None:
                    plot = ax[i,j].plot(np.array(x), values, label=k)
                    plot = plot[0]
                    plots.append(plot)
            ax[i,j].legend(handles=plots)
            ax[i,j].set_title(current_loss)

    name = kwargs.get('name', 'losses')
    if not os.path.isdir(out+'/losses'):
        out += '/losses'
        check_dir(out)
    suffix = ""
    suffix = "_"+str(partition) if partition is not None else suffix
    suffix = suffix+"_%d"%epoch if epoch is not None else suffix
    if separated:
        title = "%s_%s"%(name, loss_names[i])
        if out is not None:
            [fig[i].savefig(f"{out}/{title}{suffix}.pdf", format='pdf') for i in range(len(fig))]
    else:
        title = name
        if out is not None:
            fig.savefig(f"{out}/{title}{suffix}", format='pdf')
    return figs, axes


def plot_class_losses(dataset, model, evaluators, tasks=None, batch_size=512, partition=None, n_points=None, ids=None, epoch=None,
                      label=None, loader=None, balanced=True, preprocess=False, preprocessing=None, out=None, name=None, **kwargs):
    assert tasks
     # get plotting ids
    if partition is not None:
        dataset = dataset.retrieve(partition)
    if ids is not None:
        dataset = dataset.retrieve(ids)
    if n_points is not None:
        dataset = dataset.retrieve(np.random.permutation(len(dataset.data))[:n_points])

    classes = {}
    class_ids = {}
    full_ids = CollapsedIds()
    for t in tasks:
        class_ids[t], classes[t] = core.get_class_ids(dataset, t, balanced=balanced, split=True)
        full_ids.add(t, np.concatenate(list(class_ids[t].values())))

    Loader = loader if loader else DataLoader
    loader = Loader(dataset, batch_size, ids=full_ids.get_full_ids(), tasks = label)
    # forward
    add_args = {}
    if hasattr(model, 'prediction_params'):
        add_args['n_preds'] = model.prediction_params['n_predictions']
    add_args['epoch'] = kwargs.get('epoch')

    outs = []; targets = []
    for x,y in loader:
        if preprocess and preprocessing:
            x = preprocessing(x)
        targets.append(x)
        vae_out = model.forward(x, y=y, **add_args)
        outs.append(decudify(vae_out))
    outs = merge_dicts(outs)
    targets = torch.from_numpy(np.concatenate(targets, axis=0)).float()
    #plt.figure()
    #plt.plot(targets[10]); plt.plot(outs['x_params'].mean[10])
    #plt.savefig('caca.pdf')


    # forward!
    figs = {}; axes = {}
    for t in tasks:
        eval_dict = {}
        # obtain evaluations
        for class_name, i in class_ids[t].items():
            t_ids = full_ids.transform(i)
            out_class = recgetitem(outs, t_ids)
            eval_results = [e.evaluate(out_class, target=targets[t_ids], model=model) for e in evaluators]
            eval_dict[class_name] = eval_results
        # plot
        n_evals = len(eval_dict[class_name])
        n_rows, n_columns = core.get_divs(n_evals)
        fig, ax = plt.subplots(n_rows, n_columns)
        if n_rows == 1:
            ax = ax[np.newaxis]
        if n_columns == 1:
            ax = ax[:, np.newaxis]

        sorted_class_ids = sorted(eval_dict.keys())
        zoom = 0.9

        for i in range(n_rows):
            for j in range(n_columns):
                print(t, i,j)
                # get hash for losses
                loss_axis = {k: ax[i,j].twinx() for k in eval_results[i*n_columns+j].keys()}
                # get current losses
                values = {l: [] for l in eval_dict[list(eval_dict.keys())[0]][i*n_columns+j].keys()}
                for q, id in enumerate(sorted_class_ids):
                    current_result = eval_dict[id][i*n_columns+j]
                    # print(i, j, current_result)
                    ordered_keys = sorted(list([str(k) for k in current_result.keys()]))
                    color_map = core.get_cmap(len(ordered_keys))
                    for s, k in enumerate(ordered_keys):
                        origin = s
                        height = current_result[k]
                        # print(np.isnan(height), np.isinf(height), height)
                        if np.isnan(height) or np.isinf(height):
                            height = 0
                        # print(loss_axis.keys())
                        try:
                            loss_axis[k].bar(origin + zoom*(2*q+1)/(2*len(sorted_class_ids)), height, width = zoom/len(sorted_class_ids), color=color_map(q))
                        except KeyError:
                            pdb.set_trace()
                        loss_axis[k].set_xlim([-0.5, len(ordered_keys)])
                        loss_axis[k].set_yticks([])
                        if type(current_result[k]) in [list, tuple]:
                            for w in range(len(current_result[k])):
                                if not str(k)+"_%d"%w in values.keys():
                                    values[str(k)+'_%d'%w] = []
                                    del values[str(k)]
                                values[str(k)+'_%d'%w].append(float(current_result[k][w]))

                            loss_index = ordered_keys.index(str(k))
                            del ordered_keys[loss_index]
                            [ordered_keys.insert(loss_index+w, str(k)+"_%d"%w) for w in range(len(current_result[k]))]
                        else:
                            if not str(k) in values.keys():
                                values[str(k)] = []
                            values[k].append(current_result[k])
                ax[i,j].set_xticks([s for s in range(len(ordered_keys))])
                ax[i,j].set_xticklabels(ordered_keys, rotation=-0, ha='left', fontsize=6, minor=False)
                ax[i,j].set_yticks([])
                for loss_name, losses in values.items():
                    min_idx = np.argmin(np.array(losses)); max_idx = np.argmax(np.array(losses))
                    min_x = ordered_keys.index(loss_name) + zoom * (2 * min_idx + 1) / (2 * len(sorted_class_ids))
                    max_x = ordered_keys.index(loss_name) + zoom * (2 * max_idx + 1) / (2 * len(sorted_class_ids))
                    min_y = losses[min_idx]
                    max_y = losses[max_idx]
                    if np.isnan(min_y):
                        min_y = 0
                    if np.isnan(max_y):
                        max_y = 0
                    # print(min_idx, min_x, min_y, losses)
                    if np.log10(min_y) > 2:
                        min_y_str = "%.1e"%min_y
                    else:
                        min_y_str = "%.2f"%min_y
                    if np.log10(max_y) > 2:
                        max_y_str = "%.1e"%max_y
                    else:
                        max_y_str = "%.2f"%max_y
                    print(min_y_str, max_y_str)
                    loss_axis[loss_name].text(float(min_x), float(min_y), min_y_str,  fontsize=4, horizontalalignment='center', color=color_map(min_idx))
                    loss_axis[loss_name].text(float(max_x), float(max_y), max_y_str,  fontsize=4, horizontalalignment='center', color=color_map(max_idx))

            title = '/class_losses_%s_%s'%(t, partition)
            suffix = ""
            suffix = "_"+str(partition) if partition is not None else suffix
            suffix = suffix+"_%d"%epoch if epoch is not None else suffix
            if not out is None:
                out_dir = out + '/class_losses'
                check_dir(out_dir)
                fig.savefig(f"{out_dir}/{title}{epoch}.pdf", format="pdf")

    return figs, axes





################################################
########        MODEL ANALYSIS PLOTS
####


def plot_conv_weights(dataset, model, out=None, *args, **kwargs):
    weights = {}
    for i, current_encoder in enumerate(model.encoders):
        hidden_module = current_encoder.hidden_modules
        if issubclass(type(hidden_module), (ConvolutionalLatent, DeconvolutionalLatent)):
            layers = hidden_module.conv_module.conv_encoders
            for l in range(len(layers)):
                if issubclass(type(layers[l]), ConvLayer):
                    weights['encoder.%d.%d.weight'%(i,l)] = layers[l].conv_module.weight.data
                elif issubclass(type(layers[l]), GatedConvLayer):
                    weights['encoder.%d.%d.tanh_weight'%(i,l)] = layers[l].conv_module.weight.data
                    weights['encoder.%d.%d.sig_weight'%(i,l)] = layers[l].conv_module_sig.weight.data
                    weights['encoder.%d.%d.residual_weight'%(i,l)] = layers[l].conv_module_residual.weight.data
                    weights['encoder.%d.%d.1x1_weight'%(i,l)] = layers[l].conv_module_1x1.weight.data

    for i, current_decoder in enumerate(model.decoders):
        hidden_module = current_decoder.hidden_modules
        if issubclass(type(hidden_module), (MultiHeadConvolutionalLatent, MultiHeadDeconvolutionalLatent)):
            layers = hidden_module.conv_module.conv_encoders
            for h, head in enumerate(layers):
                layers = head.conv_encoders
                for l in range(len(layers)):
                    if issubclass(type(layers[l]), DeconvLayer):
                        weights['decoder.%d.%d.%d.weight'%(h,i,l)] = layers[l].conv_module.weight.data
                    elif issubclass(type(layers[l]), GatedDeconvLayer):
                        weights['decoder.%d.%d.%d.tanh_weight'%(h,i,l)] = layers[l].conv_module.weight.data
                        weights['decoder.%d.%d.%d.sig_weight'%(h,i,l)] = layers[l].conv_module_sig.weight.data
                        weights['decoder.%d.%d.%d.residual_weight'%(h,i,l)] = layers[l].conv_module_residual.weight.data
                        weights['decoder.%d.%d.%d.1x1_weight'%(h,i,l)] = layers[l].conv_module_1x1.weight.data
        elif issubclass(type(hidden_module), (ConvolutionalLatent, DeconvolutionalLatent)):
            layers = hidden_module.conv_module.conv_encoders
            for l in range(len(layers)):
                if issubclass(type(layers[l]), DeconvLayer):
                    weights['decoder.%d.%d.weight'%(i,l)] = layers[l].conv_module.weight.data
                elif issubclass(type(layers[l]), GatedDeconvLayer):
                    weights['decoder.%d.%d.tanh_weight'%(i,l)] = layers[l].conv_module.weight.data
                    weights['decoder.%d.%d.sig_weight'%(i,l)] = layers[l].conv_module_sig.weight.data
                    weights['decoder.%d.%d.residual_weight'%(i,l)] = layers[l].conv_module_residual.weight.data
                    weights['decoder.%d.%d.1x1_weight'%(i,l)] = layers[l].conv_module_1x1.weight.data
        out_modules = current_decoder.out_modules
        # for o in range(len(out_modules)):
        #     if issubclass(type(current_decoder.out_modules[o]), Deconvolutional):
        #     print('coucou')

    figs = []; axis = []
    if not os.path.isdir(out):
        os.makedirs(out)
    for name, kernel in weights.items():
        fig = plt.figure()
        kernel_grid = make_grid(kernel.transpose(0,2).transpose(1,2).unsqueeze(1))
        plt.imshow(kernel_grid.detach().transpose(0,2).transpose(0,1).numpy())
        figs.append(fig); axis.append(fig.axes)
        if out is not None:

            fig.savefig(out+'/%s.pdf'%name)

    return figs, axis


