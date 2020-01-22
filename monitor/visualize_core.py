#######!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:00:26 2018

@author: chemla
"""
import pdb, torch, numpy as np, random, math
from functools import reduce
import matplotlib.pyplot as plt
import itertools
from ..utils.onehot import oneHot, fromOneHot
from ..utils.misc import CollapsedIds
from sklearn.metrics import confusion_matrix
import matplotlib.patches as mpatches
from  ..import distributions as dist

#%% Various utilities for plotting


def get_cmap(n, color_map='plasma'):
    return plt.cm.get_cmap(color_map, n)

def get_class_ids(dataset, task, balanced=False, ids=None, split=False):
    '''
    returns per-ids data index relatively to given task

    :param dataset: dataset object
    :param task: task name
    :param ids: only among given ids ( default: all )
    :return:
    '''
    if dataset.classes[task].get('_length'):
        n_classes = dataset.classes[task]['_length']
    else:
        n_classes = len(list(dataset.classes.get(task).values()))
    class_ids = dict(); n_class = set()
    for meta_id in range(n_classes):
        current_ids = np.array(list(dataset.get_ids_from_class([meta_id], task, ids=ids)))
        if len(current_ids) == 0:
            continue
        class_ids[meta_id] = current_ids
        current_meta = dataset.metadata.get(task)[current_ids]
        if current_meta is not None:
            if hasattr(current_meta[0], '__iter__'):
                n_class |= set(np.concatenate(current_meta).tolist())
            else:
                n_class |= set(current_meta)

    bound = np.inf
    if balanced:
        bound = np.amin([len(a) for a in class_ids.values()])

    n_class = list(n_class)
    for i in n_class:
        if len(class_ids[i]) >  bound:
            retained_ids = np.sort(np.random.permutation(len(class_ids[i]))[:bound])
            class_ids[i] = class_ids[i][retained_ids]

    if not split:
        class_ids = np.concatenate(list(class_ids.values())).tolist()
    return class_ids, n_class






def get_divs(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    primfac = np.array(primfac)
    dims = (int(np.prod(primfac[0::2])), int(np.prod(primfac[1::2])))
    if dims[0] < dims[1]:
        dims = (dims[1], dims[0])
    return dims





# Plotting functions

def plot_mean_1d(dist, x=None, preprocess=None, preprocessing=None, axes=None, multihead=None, concat_seq=None, suffix=None, out=None, *args, **kwags):
    n_examples = dist.batch_shape[0]
    # create axes
    if axes is None:
        if len(dist.mean.shape) <= 2:
            n_rows, n_columns = get_divs(n_examples)
        elif len(dist.mean.shape) == 3:
            # is sequence
            n_rows = n_examples
            n_columns = dist.mean.shape[1]

        if preprocessing:
            fig, axes = plt.subplots(n_rows, 2*n_columns, figsize=(20, 10))
        else:
            fig, axes = plt.subplots(n_rows, n_columns, figsize=(10,10))
            if n_columns == 1:
                axes = axes[:, np.newaxis]
        if n_rows == 1:
            axes = axes[np.newaxis, :]
    else:
        fig = None

    # get distributions
    dist_mean = dist.mean.cpu().detach().numpy(); dist_mean_inv=None
    dist_variance = dist.variance.cpu().detach().numpy()
    if torch.is_tensor(x):
        x = x.cpu().detach().numpy()

    if preprocessing is not None:
        dist_mean_inv = preprocessing.invert(dist_mean)
        if preprocess:
            x, x_inv = preprocessing(x), x
        else:
            x, x_inv = x, preprocessing.invert(x)

    seq_stride = 1
    for i in range(n_rows):
        for j in range(n_columns):
            if len(dist.mean.shape) == 2:
                x_orig = x[i*n_columns+j]; x_dist = dist_mean[i*n_columns+j]
                if preprocessing:
                    x_orig_pp = x_inv[i*n_columns+j]; x_dist_pp = dist_mean_inv[i*n_columns+j]
            else:
                x_orig = x[i, j*seq_stride]; x_dist = dist_mean[i, j*seq_stride]
                if preprocessing:
                    x_orig_pp = x_inv[i, j*seq_stride]; x_dist_pp = dist_mean_inv[i, j*seq_stride]

            if preprocessing:
                if x is not None:
                    axes[i,2*j].plot(x_orig, linewidth=0.6)
                axes[i,2*j].plot(x_dist, linewidth=0.4)
                #if hasattr(dist, 'variance'):
                #    axes[i,2*j].bar(range(dist_variance.shape[1]), dist_variance[i*n_columns+j], align='edge', alpha=0.4)
                if x_inv is not None:
                    axes[i, 2*j+1].plot(x_orig_pp, linewidth=0.6)
                axes[i, 2*j+1].plot(x_dist_pp, linewidth=0.4)
                plt.tick_params(axis='y',  which='both',  bottom='off')
            else:
                if x is not None:
                    axes[i,j].plot(x_orig, linewidth=0.6)
                axes[i,j].plot(x_dist, linewidth=0.4)

    # multihead plot
    if multihead is not None:
        for k in range(len(multihead)):
            fig_m, axes_m = plt.subplots(n_rows, n_columns, figsize=(10,10))
            if len(axes_m.shape) == 1:
                axes_m = axes_m[:, np.newaxis]
            for i in range(n_rows):
                for j in range(n_columns):
                    axes_m[i, j].plot(multihead[k][i*n_columns+j].squeeze().numpy(), linewidth=0.6)
            fig.append(fig_m)
            axes.append(axes_m)

    if out is not None:
        fig.savefig(out+'.pdf', format="pdf")

    return fig, axes

def plot_mean_2d(dist, x=None, preprocessing=None, preprocess="False", multihead=None, out=None, *args, **kwargs):
    n_examples = dist.batch_shape[0]
    n_rows, n_columns = get_divs(n_examples)
    has_std = hasattr(dist, 'stddev')
    if x is None:
        fig, axes = plt.subplots(n_rows, n_columns)
        if has_std:
            fig_std, axes_std = plt.subplots(n_rows, n_columns)
    else:
        fig, axes = plt.subplots(n_rows, 2 * n_columns)
        if has_std:
            fig_std, axes_std = plt.subplots(n_rows, 2*n_columns)

    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
        if has_std:
            axes_std = axes_std[np.newaxis, :]

    dist_mean = dist.mean.cpu().detach().numpy()
    if has_std:
        dist_std = dist.stddev.cpu().detach().numpy()

    if preprocess:
        assert preprocessing, 'if preprocess is on then give a preprocessing object'
        x = preprocessing.invert(x.cpu().detach().numpy())
        dist_mean = preprocessing.invert(dist_mean)

    for i in range(n_rows):
        for j in range(n_columns):
                if x is not None:
                    axes[i,2*j].imshow(x[i*n_columns+j], aspect='auto')
                    axes[i,2*j].set_title('data')
                    axes[i,2*j+1].imshow(dist_mean[i*n_columns+j], aspect='auto')
                    axes[i,2*j+1].set_title('reconstruction')
                else:
                    axes[i,j].set_title('data')
                    axes[i,j+1].imshow(dist_mean[i*n_columns+j], aspect='auto')
                    axes[i,j+1].set_title('reconstruction')
                if hasattr(dist, "stddev"):
                    if x is not None:
                        axes_std[i,2*j].imshow(x[i*n_columns+j], aspect='auto')
                        axes_std[i,2*j].set_title('data')
                        axes_std[i,2*j+1].imshow(dist_std[i*n_columns+j],vmin=0, vmax=1, aspect='auto')
                        axes_std[i,2*j+1].set_title('reconstruction')
                    else:
                        axes_std[i,j].set_title('data')
                        axes_std[i,j+1].imshow(dist_std[i*n_columns+j], vmin=0, vmax=1, aspect='auto')
                        axes_std[i,j+1].set_title('reconstruction')

    if multihead is not None:
        fig = [fig]; axes = [axes];
        fig_m, axes_m = plt.subplots(n_rows, n_columns, figsize=(10,10))
        if len(axes_m.shape) == 1:
            axes_m = axes_m[:, np.newaxis]
        for k in range(len(multihead)):
            for i in range(n_rows):
                for j in range(n_columns):
                    axes_m[i, j].imshow(multihead[k][i*n_columns+j].squeeze().numpy(), aspect="auto")
            fig.append(fig_m)
            axes.append(axes_m)

    if out is not None:
        fig.savefig(out+".pdf", format="pdf")
        if has_std:
            fig_std.savefig(out+"_std.pdf", format="pdf")

    return fig, axes


def plot_dirac(x, *args, **kwargs):
    x = dist.Normal(x, torch.zeros_like(x))
    return plot_mean(x, *args, **kwargs)


def plot_mean(x, target=None, preprocessing=None, axes=None, *args, is_sequence=False, **kwargs):
    is_sequence = is_sequence and x.mean.shape[1] > 1
    if type(x) == dist.Normal:
        x = type(x)(x.mean.squeeze(), x.stddev.squeeze())
    else:
        x = type(x)(x.mean.squeeze())
    target = target.squeeze()
    if is_sequence:
        fig = [None]*x.batch_shape[0]; axes = [None]*x.batch_shape[0]
        for ex in range(x.batch_shape[0]):
            if len(x.batch_shape) <= 3:
                fig[ex], axes[ex] = plot_mean_1d(x[ex], x=target[ex], preprocessing=preprocessing, *args, **kwargs)
            elif len(x.batch_shape) == 4:
                fig, axes = plot_mean_2d(x, x=target, preprocessing=preprocessing,  *args, **kwargs)
    else:
        if len(x.batch_shape) <= 2 + is_sequence:
            fig, axes = plot_mean_1d(x, x=target, preprocessing=preprocessing, axes=axes, *args, **kwargs)
        elif len(x.batch_shape) == 3 + is_sequence:
            fig, axes = plot_mean_2d(x, x=target, preprocessing=preprocessing, axes=axes, *args, **kwargs)
    return fig, axes


def plot_probs(x, target=None, preprocessing=None, *args, **kwargs):
    n_examples = x.batch_shape[0]
    n_rows, n_columns = get_divs(n_examples)
    fig, axes = plt.subplots(n_rows, n_columns) if len(target.shape) == 2 else plt.subplots(n_rows, 2*n_columns)
    if not issubclass(type(axes), np.ndarray):
        axes = np.array(axes)[np.newaxis]
    if axes.ndim == 1:
        axes = axes[:, np.newaxis]
    for i in range(n_rows):
        for j in range(n_columns):
            if target is not None:
                if len(target.shape) < 2:
                    target = oneHot(target, dist.batch_shape[1])
                if torch.is_tensor(target):
                    target = target.cpu().detach().numpy()
                axes[i,j].plot(target[i*n_columns+j])
            if len(target.shape) == 2:
                probs = x.probs[i*n_columns+j]
                if torch.is_tensor(probs):
                    probs = probs.cpu().detach().numpy()
                axes[i,j].plot(probs, linewidth=0.5)
                plt.tick_params(axis='y',  which='both',  bottom='off')
            elif len(x.shape) == 3:
               raise NotImplementedError
    return fig, axes


plotting_hash = {torch.Tensor: plot_dirac,
                 dist.Normal: plot_mean,
                 dist.Bernoulli: plot_mean,
                 dist.Categorical: plot_probs}

def plot_distribution(dists, *args, **kwargs):
    if issubclass(type(dists), list):
        return [plot_distribution(dists[i], *args, **kwargs) for i in range(len(dists))]
    if not type(dists) in plotting_hash.keys():
        raise TypeError("error in plot_distribution : don't have a plot callback for type %s"%type(dists))
    fig, ax = plotting_hash[type(dists)](dists, *args, **kwargs)
    return fig, ax


def plot_latent_path(zs, data, synth, reduction=None):
    fig = plt.figure()
    grid = plt.GridSpec(1, 4, hspace=0.2, wspace=0.2)
        
    if not reduction is None:
        zs = reduction.transform(zs)
        
    gradient = get_cmap(zs.shape[0])
    if zs.shape[1] == 2:
        ax = fig.add_subplot(grid[:2])
        ax.plot(zs[:,0],zs[:,1], c=gradient(np.arange(zs.shape[0])))
    elif zs.shape[1] == 3:
        ax = fig.add_subplot(grid[:2], projection='3d', xmargin=1)
        for i in range(zs.shape[0]-1):
            ax.plot([zs[i,0], zs[i+1, 0]], [zs[i,1], zs[i+1, 1]], [zs[i,2], zs[i+1, 2]], c=gradient(i))

    ax = fig.add_subplot(grid[2])
    ax.imshow(data, aspect='auto')
    ax = fig.add_subplot(grid[3])
    ax.imshow(synth, aspect='auto')
    return fig, fig.axes


def plot(current_z, *args, **kwargs):
    if current_z.shape[1] == 2:
        fig, ax = plot_2d(current_z, *args, **kwargs)
    elif current_z.shape[1] >= 3:
        fig, ax = plot_3d(current_z, *args, **kwargs)
    return fig, ax


def plot_2d(current_z, meta=None, var=None, classes=None, class_ids=None, class_names=None, cmap='plasma',
            sequence=None, shadow_z=None, centroids=None, legend=True):
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()

    if meta is None:
        meta = np.zeros((current_z.shape[0]))
        cmap = get_cmap(0, color_map=cmap)
        cmap_hash = {0:0}
    else:
        cmap = get_cmap(len(classes), color_map=cmap)
        cmap_hash = {classes[i]:i for i in range(len(classes))}

    current_alpha = 0.06 if (centroids and not meta is None) else 1.0
    current_var = var if not var is None else np.ones(current_z.shape[0])
    current_var = (current_var - current_var.mean() / np.abs(current_var).max())+1
    meta = meta.astype(np.int)

    # plot
    if sequence:
        if shadow_z is not None:
            for i in range(shadow_z.shape[0]):
                ax.plot(shadow_z[i, :, 0], shadow_z[i, :,1], c=np.array([0.8, 0.8, 0.8, 0.4]))
        for i in range(current_z.shape[0]):
            ax.plot(current_z[i, :, 0], current_z[i, :,1], c=cmap(cmap_hash[meta[i]]), alpha = current_alpha)
            ax.scatter(current_z[i,0,0], current_z[i,0,1], c=cmap(cmap_hash[meta[0]]), alpha = current_alpha, marker='o')
            ax.scatter(current_z[i,-1,0], current_z[i,-1,1], c=cmap(cmap_hash[meta[0]]), alpha = current_alpha, marker='+')
    else:
        cs = np.array([cmap_hash[m] for m in meta])
        ax.scatter(current_z[:, 0], current_z[:,1], c=cs, alpha = current_alpha, s=current_var)
    # make centroids
    if centroids and not meta is None:
        for i, cid in class_ids.items():
            centroid = np.mean(current_z[cid], axis=0)
            ax.scatter(centroid[0], centroid[1], centroid[2], s = 30, c=cmap(classes[i]))
            ax.text(centroid[0], centroid[1], centroid[2], class_names[i], color=cmap(classes[i]), fontsize=10)
    # make legends
    if legend and not meta is None and not classes is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cmap_hash[cl]), label=str(class_names[cl]))
            handles.append(patch)
        ax.legend(handles=handles, loc='upper left', borderaxespad=0.)

    return fig, ax


def plot_3d(current_z, meta=None, var=None, classes=None, class_ids=None, class_names=None, cmap='plasma', sequence=False, centroids=None, legend=True, shadow_z=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection='3d')
    
    if meta is None:
        meta = np.zeros((current_z.shape[0]))
        cmap = get_cmap(0, color_map=cmap)
        cmap_hash = {0:0}
    else:
        cmap = get_cmap(0, color_map=cmap) if classes is None else get_cmap(len(classes), color_map=cmap)
        cmap_hash = {None:None} if classes is None else {classes[i]:i for i in range(len(classes))}

    current_alpha = 0.06 if (centroids and not meta is None) else 1.0
    current_var = var if not var is None else np.ones(current_z.shape[0])
    current_var = (current_var - current_var.mean() / np.abs(current_var).max())+1
    meta = meta.astype(np.int)

    # plot
    if sequence:
        if shadow_z is not None:
            for i in range(shadow_z.shape[0]):
                ax.plot(shadow_z[i, :, 0], shadow_z[i, :,1], shadow_z[i, :,2], c=np.array([0.8, 0.8, 0.8, 0.4]))
        for i in range(current_z.shape[0]):
            ax.plot(current_z[i, :, 0], current_z[i, :,1],current_z[i, :,2], c=cmap(cmap_hash[meta[i]]), alpha = current_alpha)
            ax.scatter(current_z[i,0,0], current_z[i,0,1],current_z[i,0,2], c=cmap(cmap_hash[meta[0]]), alpha = current_alpha, marker='o')
            ax.scatter(current_z[i,-1,0], current_z[i,-1,1],current_z[i,-1,2], c=cmap(cmap_hash[meta[0]]), alpha = current_alpha, marker='+')
    else:
        cs = np.array([cmap_hash[m] for m in meta])
        if current_z.shape[1]==2:
            ax.scatter(current_z[:, 0], current_z[:,1], np.zeros_like(current_z[:,0]), c=cs, alpha = current_alpha, s=current_var)
        else:
            ax.scatter(current_z[:, 0], current_z[:,1], current_z[:, 2], c=cs, alpha = current_alpha, s=current_var)
    # make centroids
    if centroids and not meta is None:
        for i, cid in class_ids.items():
            centroid = np.mean(current_z[cid], axis=0)
            ax.scatter(centroid[0], centroid[1], centroid[2], s = 30, c=cmap(classes[i]))
            ax.text(centroid[0], centroid[1], centroid[2], class_names[i], color=cmap(classes[i]), fontsize=10)
    # make legends   
    if legend and not meta is None and not classes is None:
        handles = []
        for cl in classes:
            patch = mpatches.Patch(color=cmap(cmap_hash[cl]), label=str(class_names[cl]))
            handles.append(patch)
        ax.legend(handles=handles, loc='upper left', borderaxespad=0.)
        
    return fig, ax




##############################
######   confusion matirices

def plot_confusion_matrix(confusion_matrices, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not issubclass(type(confusion_matrices), list):
        confusion_matrices = [confusion_matrices]
    if not issubclass(type(classes), list):
        classes = [classes]

    fig = plt.figure(figsize=(16,8))
    
    for i, cm in enumerate(confusion_matrices):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
            
        print(cm)
        
        ax = fig.add_subplot(1, len(confusion_matrices), i+1)
        cmap = get_cmap(max(list(classes[i].keys())), 'Blues')
        img = ax.imshow(cm, cmap=cmap)
        
        ax.set_title(title)
#        plt.colorbar(img, ax=ax)
        tick_marks = np.arange(len(classes[i]))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels([classes[i][n] for n in range(max(list(classes[i].keys())))], rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels([classes[i][n] for n in range(max(list(classes[i].keys())))])

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        
    return fig


def make_confusion_matrix(false_labels, true_labels, classes):
    if not issubclass(type(false_labels), list):
        false_labels = [false_labels]
    if not issubclass(type(false_labels), list):
        true_labels = [true_labels]
    if not issubclass(type(classes), list):
        classes = [classes]
        
    assert len(false_labels) == len(true_labels) == len(classes)      
    cnf_matrices = []
        
    for i, fl in enumerate(false_labels):
        tl = true_labels[i]
        if fl.ndim == 2:
            fl = fromOneHot(fl)
        if tl.ndim == 2:
            tl = fromOneHot(tl)
        cnf_matrices.append(confusion_matrix(fl, tl))
        
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    fig = plot_confusion_matrix(cnf_matrices, classes=classes, title='Confusion matrix, without normalization')
    return fig
    


