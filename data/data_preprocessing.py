#'a!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:26:51 2018

@author: chemla
"""
import pdb
import numpy as np
import torch
from ..utils.onehot import oneHot, fromOneHot

eps = 1e-3


### ABSTRACT PRERPROCESSING CLASS

class Preprocessing(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x, *args, **kwargs):
        return x

    def invert(self, x):
        return x

    def scale(self, data):
        pass


### AUDIO PREPROCESSING OBJECTS

class Normalize(object):
    def __init__(self, dataset=None, norm_type="minmax", mode="bipolar"):
        super(Normalize, self).__init__()
        self.norm_type = norm_type
        self.mode = mode
        if dataset is not None:
            self.scale(dataset.data)

    @staticmethod
    def get_stats(data, norm_type="gaussian", mode="bipolar"):
        if norm_type == "minmax":
            if mode == "bipolar":
                if torch.is_tensor(data):
                    mean = (torch.max(data) - torch.sign(torch.min(data))*torch.min(data)) / 2
                    max = torch.max(torch.abs(data - mean))
                else:
                    mean = (np.max(data) - np.sign(np.min(data))*np.min(data)) / 2
                    max = np.amax(np.abs(data - mean))
            elif mode == "unipolar":
                if torch.is_tensor(data):
                    mean = torch.min(data)
                    max = torch.max(torch.abs(data))
                else:
                    mean = np.amin(data)
                    max = np.amax(np.abs(data))

        elif norm_type == "gaussian":
            if torch.is_tensor(data):
                mean = torch.mean(data)
                max = torch.std(data)
            else:
                mean = np.mean(data)
                max = np.std(data)
        return mean, max

    def scale(self, data):
        if issubclass(type(data), list):
            stats = np.array([self.get_stats(d, self.norm_type, self.mode) for d in data])
            if self.norm_type == "minmax":
                self.mean = np.mean(stats[:,0]); self.max = np.amax(stats[:,1]-self.mean+1e-2)
            else:
                self.mean = np.mean(stats[:, 0]);
                # recompose overall variance from element-wise ones
                n_elt = [np.array(np.cumprod(list(x.shape))[1]) for x in data]
                std_unscaled = ((stats[:,1]**2) * (np.array(n_elt) - 1)) / (np.cumsum(np.array(n_elt), 0)[-1] - 1)
                self.max = np.sqrt(np.sum(std_unscaled))
        else:
            self.mean, self.max = self.get_stats(data, self.norm_type, self.mode)

    def __call__(self, x):
        out = (x - self.mean) / self.max
        if self.mode == "unipolar":
            out = np.clip(out, eps, None)
        return out

    def invert(self, x):
        return x * self.max + self.mean

class LogNormalize(Normalize):
    def scale(self, data):
        if type(data)==list:
            data = [np.log(np.clip(data, eps, None)) for d in data]
        else:
            data = np.log(np.clip(data, eps, None))
        super(LogNormalize, self).scale(data)

    def __call__(self, x):
        return super(LogNormalize, self).__call__(np.log(np.clip(x, eps, None)))

    def invert(self, x):
        return np.exp(super(LogNormalize, self).invert(x))


class Log1pNormalize(Normalize):
    def scale(self, data):
        if type(data)==list:
            data = [np.log1p(d) for d in data]
        else:
            data = np.log1p(data)
        super(Log1pNormalize, self).scale(data)

    def __call__(self, x):
        return super(Log1pNormalize,self).__call__(np.log1p(np.clip(x, eps, None)))

    def invert(self, x):
        return np.exp(super(Log1pNormalize, self).invert(x) - 1)


class Magnitude(object):
    log_threshold = 1e-3
    log1p_threshold = 7.0
    def __repr__(self):
        rep = "<preprocessing Magnitude with pp: %s, pre_norm: %s, post_norm:%s, center:%s>" % (self.preprocessing, self.pre_norm, self.post_norm, self.center)
        return rep

    def __init__(self, preprocessing='none', pre_norm='std', post_norm='std', shrink=1, center=False, log_threshold = None):
        super(Magnitude, self).__init__()
        self.preprocessing = preprocessing
        self.pre_norm = pre_norm; self.post_norm = post_norm
        self.preMax = None; self.postMax = None; self.postMean = None;
        self.center = center
        self.shrink = shrink
        if log_threshold:
            self.log_threshold = log_threshold

    def preprocess(self, data):
        if self.preprocessing == "log":
            data[data < self.log_threshold] = self.log_threshold
            return np.log(data)
        elif self.preprocessing == "log1p":
            return np.log1p(data)
        elif self.preprocessing == "tanh":
            return np.tanh(data)
        else:
            return data

    def scale(self, data):
        if issubclass(type(data), list) or issubclass(type(data), tuple):
            # concatenate among batch dimension
            data = np.concatenate(data, axis=0)

        data = np.abs(data)
        # get pre-normalization
        if self.pre_norm == "std":
            self.preMax = np.std(data)
            data = data / self.preMax
        elif self.pre_norm == "max":
            self.preMax = np.max(data)
            data = data / self.preMax

        # apply preprocessing
        if self.preprocessing != "none":
            data = self.preprocess(data)

        # get post_normalization
        if self.center:
            self.postMean = np.mean(data)
            data = data - self.postMean

        if self.post_norm == "std":
            self.postMax = np.std(data)*self.shrink
        elif self.post_norm == "max":
            self.postMax = np.max(data)

    def __call__(self, data, normalize=True):
        if issubclass(type(data), list):
            return [self(x) for x in data]
        new_data = np.abs(data)
        if self.preMax is not None:
            new_data = new_data / self.preMax
        new_data = self.preprocess(new_data)
        if self.postMean is not None:
            new_data = new_data - self.postMean
        if self.postMax is not None:
            new_data = new_data / self.postMax
        return new_data

    def invert(self, data):
        new_data = data
        if self.postMax is not None:
            new_data = new_data * self.postMax
        if self.postMean is not None:
            new_data = new_data + self.postMean

        if self.preprocessing == 'log':
            if torch.is_tensor(new_data):
                new_data = new_data.exp()
            else:
                new_data = np.exp(new_data)
            new_data[new_data < self.log_threshold] = 0
        elif self.preprocessing == 'log1p':
            if torch.is_tensor(new_data):
                new_data = torch.clamp(new_data, None, self.log1p_threshold)
                new_data =  new_data.exp() - 1
            else:
                new_data[new_data > self.log1p_threshold] = self.log1p_threshold
                new_data = np.exp(new_data) - 1
        elif self.preprocessing == 'tanh':
            new_data = new_data.atanh()

        if self.preMax is not None:
            new_data = new_data * self.preMax

        return new_data


class MuLaw(object):
    """
    Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1
    Args:
        quantization_channels (int): Number of channels. default: 256
    """

    def __init__(self, quantization_channels=256, normalize=False):
        self.qc = 2**quantization_channels
        self.normalize = normalize
        self.maxData = None

    def scale(self, data):
        if issubclass(type(data), list) or issubclass(type(data), tuple):
            # concatenate among batch dimension
            data = np.concatenate(data, axis=0)
        self.maxData = np.amax(np.abs(data))

    def __call__(self, x):
        """
        Args:
            x (FloatTensor/LongTensor or ndarray)
        Returns:
            x_mu (LongTensor or ndarray)
        """
        if self.normalize:
            x = x / self.maxData

        mu = self.qc - 1.
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
            n = x_mu.shape[0]
            t = np.zeros((n, self.qc, x_mu.shape[2]))
            for i in range(n):
                for j in range(x_mu.shape[2]):
                    t[i,x_mu[i,0,j],j] = 1
            x_mu = t

        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            mu = torch.FloatTensor([mu])
            x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
            n = x_mu.shape[0]
            t = torch.zeros((n, self.qc, x_mu.shape[2]), requires_grad=x.requires_grad)
            for i in range(n):
                for j in range(x_mu.shape[2]):
                    t[i,x_mu[i,0,j],j] = 1
            x_mu = t

        return x_mu


    def invert(self, x_mu):

        mu = self.qc - 1.
        if isinstance(x_mu, np.ndarray):

            t = np.zeros((x_mu.shape[0], x_mu.shape[2]))
            for i in range(x_mu.shape[0]):
                for j in range(x_mu.shape[2]):
                    t[i,j] = np.where(x_mu[i,:,j] == 1)[0][0]
            x_mu = t
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            t = np.zeros((x_mu.shape[0], x_mu.shape[2]))
            for i in range(x_mu.shape[0]):
                for j in range(x_mu.shape[2]):
                    t[i,j] = np.where(x_mu[i,:,j] == 0)[0]
            x_mu = t
            if isinstance(x_mu, torch.LongTensor):
                x_mu = x_mu.float()
            mu = torch.FloatTensor([mu])
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu

        if self.normalize:
            x = x * self.maxData

        return x.unsqueeze(1)


### SYMBOLIC PREPROCESSINGS


class OneHot(Preprocessing):
    def __repr__(self):
        return "OneHot(classes:%s, is_sequence: %s, make_channels:%s)"%(self.classes, self.is_sequence, self.make_channels)

    def __init__(self, classes=None, is_sequence=False, make_channels=False, flatten=False):
        self.classes = classes
        self.hashes = None
        if classes is not None:
            self.hashes = {k:k for k in range(classes)}
        self.is_sequence = is_sequence
        self.flatten = flatten
        self.make_channels = make_channels

    def get_hash(self, data):
        min_label = np.min(data)
        max_label = np.max(data)
        hash = {}; curHash = 0
        for i in range(min_label, max_label+1):
            is_present = np.sum(data == i) > 0
            if is_present:
                hash[int(curHash)] = i
                curHash += 1
        return curHash, hash

    def scale(self, data):
        if issubclass(type(data), (list, tuple)):
            self.classes = []; self.hashes = []
            for d in data:
                c, h = self.get_hash(d)
                self.classes.append(c)
                self.hashes.append(h)
        else:
            c, h = self.get_hash(d)
            self.classes = c
            self.hashes = h

    def __call__(self, x, classes=None, hashes=None, *args, **kwargs):
        classes = classes or self.classes
        hashes = hashes or self.hashes
        if issubclass(type(x), (list, tuple)):
            t = []
            for i in range(len(x)):
                t.append(self(x[i], classes=classes[i], hashes=hashes[i]))
            return t

        for h, k in hashes.items():
            if issubclass(type(x), np.ndarray):
                np.place(x, x==k, [h])
            elif torch.is_tensor(x):
                x = torch.where(x==k, torch.full_like(x, h), x)

        if issubclass(type(x), np.ndarray):
            np.place(x, 1-np.isin(x, np.array(list(hashes.keys()))), -1)
        elif torch.is_tensor(x):
            false_ids = filter(lambda i: not i in x, torch.arange(x.min(), x.max()))
            for f in false_ids:
                x = torch.where(x==f, x, torch.full_like(x, -1))
        t =  oneHot(x, dim=classes, is_sequence=self.is_sequence)
        if self.make_channels:
            t = t.transpose(-1, -2)
        if self.flatten:
            t = t.reshape(*t.shape[:-2], t.shape[-2]*t.shape[-1])
        return t

    def invert(self, x, hashes=None, original_size=None):
        hashes = hashes or self.hashes
        if issubclass(type(x), (list, tuple)):
            t = []
            for i in range(len(x)):
                t.append(self.invert(x[i], hashes=hashes[i]))
            return t

        if self.flatten:
            n_seq = x.shape[-1] // self.classes
            x = x.reshape(*x.shape[:-1], self.classes, n_seq)
        if self.make_channels:
            x = x.transpose(-2, -1)

        out = fromOneHot(x)
        for h, k in hashes.items():
            if issubclass(type(out), np.ndarray):
                np.place(out, out==h, [k])
            else:
                out = torch.where(out==h, torch.full_like(out, k), out)
        return out


