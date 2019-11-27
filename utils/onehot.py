#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:46:21 2018

@author: chemla
"""
import pdb
from numbers import Number
import torch, numpy as np
from torch.autograd import Variable

def oneHot(labels, dim, is_sequence=False):
    if isinstance(labels, Number):
        t = np.zeros((1, dim))
        t[0, int(labels)] = 1
    else:
        if len(labels.shape) <= 2:
            if issubclass(type(labels), np.ndarray):
                n = labels.shape[0]
                t = np.zeros((n, dim))
                for i in range(n):
                    if labels[i] == -1:
                        continue
                    t[i, int(labels[i])] = 1
            elif torch.is_tensor(labels):
                n = labels.size(0)
                t = torch.zeros((n, dim), device=labels.device)
                for i in range(n):
                    if labels[i] == -1:
                        continue
                    t[i, int(labels[i])] = 1
                t.requires_grad_(labels.requires_grad)
            else:
                raise Exception('type %s is not recognized by oneHot function'%type(labels))
        elif len(labels.shape) == 3:
            orig_shape = labels.shape[:2]
            labels = labels.reshape(labels.shape[0]*labels.shape[1], *labels.shape[2:])
            t = np.concatenate([oneHot(labels[i], dim)[np.newaxis] for i in range(labels.shape[0])], axis=0)
            t = t.reshape((orig_shape[0], orig_shape[1], *t.shape[1:]))
    return t

def fromOneHot(vector, is_sequence=False):
    axis = 2 if is_sequence else 1
    if issubclass(type(vector), np.ndarray):
        ids = np.argmax(vector, axis=axis)
        return ids
    elif issubclass(type(vector), torch.Tensor):
        return torch.argmax(vector, dim=axis)
    else:
        raise TypeError('vector must be whether a np.ndarray or a tensor.')
