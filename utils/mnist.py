#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:45:47 2017

@author: chemla
"""
import pdb
from numpy.random import permutation
from ..data.data_generic import Dataset
from torchvision import datasets
from numpy import array, reshape

class MNISTDataset(Dataset):
    def __init__(self, options={}, binarized=False, flatten=True, n_images=None):
        super(MNISTDataset, self).__init__({'dataPrefix':'variational-synthesis/data/toys/mnist/raw'})
        raw_dataset = datasets.MNIST(self.dataPrefix, train=True, download=True)
        self.data = raw_dataset.train_data.float().div_(255).numpy()
        if n_images:
            ids = permutation(self.data.shape[0])[:n_images]
            self.data = self.data[ids]
            self.partitions={'train':array(range(5*(n_images//6))), 'test':array(range(5*(n_images//6), n_images))}
        else:
            self.partitions={'train':array(range(50000)), 'test':array(range(50000, 60000))}
        if flatten:
            self.data = reshape(self.data, (self.data.shape[0], 784))
        self.metadata = {"class":raw_dataset.train_labels.numpy()}
        self.classes = {'class':{'_length':10}}
        for i in range(10):
            self.classes['class'][str(i)]=i
        self.original_size = (32,32)
        if binarized:
            self.data[self.data<0.5]=0
            self.data[self.data>=0.5]=1
        self.tasks = ['class']
        
