#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:14:16 2018

@author: chemla
"""

import numpy as np
import torch
import pdb
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def plot_transfers(datasets, models, transformation, n_iter=1, label=None, out=None, ids=None):
    figs = []; axes = []
    ids = np.random.permutation(datasets[0].data.shape[0])[:n_iter]
    
    if not label is None:
        if not issubclass(type(label), list):
            label = [label]
        if not label is None:
            current_metadata = [{l: datasets[0].metadata.get(l)[ids] for l in label},
                         {l: datasets[1].metadata.get(l)[ids] for l in label}]
    else:
        current_metadata = None
    
    with torch.no_grad():
        if issubclass(type(datasets[0].data), list):
            len_1 = len(datasets[0].data)
            x_1 = [d[ids] for d in datasets[0].data]
        else:
            len_1 = 1
            x_1 = [datasets[0].data[ids]]
        if issubclass(type(datasets[1].data), list):
            len_2 = len(datasets[1].data)
            x_2 = [d[ids] for d in datasets[1].data]
        else:
            len_2 = 1
            x_2 = [datasets[1].data[ids]]
            
        out_1 = models[0].forward(x_1, y=current_metadata)
        zs_1 = out_1['z_params_enc'][-1][0]
        
        out_2 = models[1].forward(x_2, y=current_metadata)
        zs_2 = out_2['z_params_enc'][-1][0]
        
        outs_transf_1 = models[0].decode(zs_2)[0]['out_params']
        outs_transf_2 = models[1].decode(zs_1)[0]['out_params']
        
        if issubclass(type(out_1['x_params']), tuple):
            out_1['x_params'] = [out_1['x_params']]
            outs_transf_1 = [outs_transf_1]
        if issubclass(type(out_1['x_params']), tuple):
            out_2['x_params'] = [out_2['x_params']]
            outs_transf_2 = [outs_transf_2]

    for i, current_id in enumerate(ids):
        fig = plt.figure(figsize=(15,4))
        grid = plt.GridSpec(2, max(len_1, len_2) + 2*(len_1+len_2)+3, hspace=0.2, wspace=0.2)
        
#         plot whole latent space
        ax2 = fig.add_subplot(grid[:, max(len_1,len_2):max(len_1,len_2)+2], projection='3d', xmargin=1)
        current_zs = torch.cat((zs_1[i].unsqueeze(0), zs_2[i].unsqueeze(0)), 0).detach().numpy()
        colors = np.array([mcolors.to_rgba('b'), mcolors.to_rgba('r')])
        ax2.scatter(current_zs[:, 0], current_zs[:, 1], current_zs[:, 2], c=colors)
#        plt.subplots_adjust(right = 1.5)

#        pdb.set_trace()
        if not issubclass(type(out_1['x_params']), list):
            out_1['x_params'] = [out_1['x_params']]
            outs_transf_1 = [outs_transf_1]
        for j in range(len_1):
            ax1 = fig.add_subplot(grid[0, j])
            ax1.plot(x_1[j][i], 'b'); 
            ax1.set_xticks([]); 
        
            ax2 = fig.add_subplot(grid[0, max(len_1,len_2)+3+j])
            ax2.set_xticks([]); 
            ax2.plot(out_1['x_params'][j][0][i].detach().numpy(), 'b');
            
            ax3 = fig.add_subplot(grid[1, max(len_1,len_2) + (len_1+len_2)+3+j])
            ax3.plot(outs_transf_1[j][0][i].detach().numpy(), 'b');
            
            if j==0:
                ax1.set_ylabel('data / domain 1')
                ax2.set_ylabel('reconstructions')
                ax3.set_ylabel('translations')
            
        if not issubclass(type(out_2['x_params']), list):
            out_2['x_params'] = [out_2['x_params']]
            outs_transf_2 = [outs_transf_2]
            
        for j in range(len_2):
            ax1 = fig.add_subplot(grid[1, j])
            ax1.set_xticks([]); 
            ax1.plot(x_2[j][i], 'r')
        
            ax2 = fig.add_subplot(grid[1, max(len_1,len_2)+3+j])
            ax2.set_xticks([]); 
            ax2.plot(out_2['x_params'][j][0][i].detach().numpy(), 'r');
            
            ax3 = fig.add_subplot(grid[0, max(len_1,len_2) + (len_1+len_2)+3+j])
            ax3.plot(outs_transf_2[j][0][i].detach().numpy(), 'r');
            
            if j==0:
                ax1.set_ylabel('data / domain 2')
                ax2.set_ylabel('reconstructions')
                ax3.set_ylabel('translations')

        
        fig.tight_layout(pad=1, w_pad=1, h_pad=1.0)
        
        figs.append(fig)
        if not out is None:
            fig.savefig(out+'_%d.pdf'%current_id)
            
    return fig, axes
        

#def plot_transfer_confusion_matrix()
        
