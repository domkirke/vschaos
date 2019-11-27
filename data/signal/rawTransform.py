#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:32:09 2018

@author: chemla
"""
import pdb
import numpy as np, torch, os
import librosa
import resampy
import cProfile

class DownmixMono(object):
    """
    Downmix any stereo signals to mono

    Inputs:
        tensor (Tensor): Tensor of audio of size (Samples x Channels)
    Returns:
        tensor (Tensor) (Samples x 1):
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        if tensor.shape[1] > 1:
            tensor = np.mean(tensor, axis=1, keepdims=True)
        return tensor


class MuLawEncoding(object):
    """
    Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1
    Args:
        quantization_channels (int): Number of channels. default: 256
    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x):
        """
        Args:
            x (FloatTensor/LongTensor or ndarray)
        Returns:
            x_mu (LongTensor or ndarray)
        """
        mu = self.qc - 1.
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            mu = torch.FloatTensor([mu])
            x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
        return x_mu


class MuLawExpanding(object):
    """
    Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.
    Args:
        quantization_channels (int): Number of channels. default: 256
    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x_mu):
        """
        Args:
            x_mu (FloatTensor/LongTensor or ndarray)
        Returns:
            x (FloatTensor or ndarray)
        """
        mu = self.qc - 1.
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            if isinstance(x_mu, torch.LongTensor):
                x_mu = x_mu.float()
            mu = torch.FloatTensor([mu])
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x


def keep_silence(data, keep_rate=0.2, noise_threshold=5e-4):
    rmse = np.mean(librosa.feature.rmse(data))
    if rmse < noise_threshold:
        keep = (np.random.rand(1) < keep_rate)[0]
    else:
        keep = True
    return keep


def splitRawSignal(sig, options):
    resampleTo = options.get('resampleTo', 22050)
    grainSize = options.get('chunkSize', 0)
    grainHop = options.get('chunkHop', grainSize / 2)
    muLaw = options.get('muLaw', 0)
    silence_keep = options.get('silenceKeep', 0.2)

    '''
    if sr != resampleTo:
        print('resampling')
        sig = resampy.resample(sig, sr, resampleTo)
    '''
    idxs = None
    if grainSize <= -1:
        currentTransform = sig[np.newaxis]
        idxs = (0.0, sig.shape[0] / resampleTo, resampleTo)
    else:
        sig_size = sig.shape[0]
        n_chunks = (sig_size-grainSize)//grainHop
        currentTransform = np.zeros((n_chunks, grainSize))
        items_to_delete = []; idxs = []
        for n in range(n_chunks):
            currentTransform[n, :] = sig[n*grainHop:n*grainHop+grainSize]
            if librosa.feature.rmse(currentTransform[n, :]).mean() < 1e-4:
                items_to_delete.append(n)
            idxs.append(( (n * grainHop) / resampleTo, grainSize / resampleTo, resampleTo))
        currentTransform = np.delete(currentTransform, np.array(items_to_delete), axis=0)
        for i in reversed(items_to_delete):
            del idxs[i]
    if muLaw > 0:
        encoder = MuLawEncoding(muLaw)
        currentTransform = encoder(currentTransform)
    if options.get('channel_dim'):
        currentTransform = np.expand_dims(currentTransform, 1)

    #return currentTransform, idxs
    currentTransform = [currentTransform[i] for i in range(currentTransform.shape[0])]
    currentTransform = list(filter(lambda x: keep_silence(x, keep_rate=silence_keep), currentTransform))
    return currentTransform


def concatSignal(sig, window_size):
    window = torch.hann_window(window_size).unsqueeze(0).numpy()
    sig = (window.T * sig.squeeze().T).T
    sig_concat = np.zeros((1,int((sig.shape[0]/2+0.5)*window_size)))

    for i in range(sig.shape[0]):
        sig_concat[0, i*int(window_size/2):(i*int(window_size/2)+window_size)] += sig[i]

    return sig_concat.squeeze()



def parseRawData(file, output_path, options, baseroot=None):
    resampleTo = options.get('resampleTo', 22050)
    chunk_size = options.get('chunkSize', 0.2)
    chunk_overlap = options.get('chunkOverlap', 0.1)
    silence_keep = options.get('silenceKeep')
    files = []
    for f in file:
        print('exporting %s...'%f)
        currentTransform, _ = importRawSignal(f, {'resampleTo': resampleTo, 'muLaw': options.get('muLaw')})
        split_length = int(chunk_size * resampleTo)
        split_hop = int(chunk_overlap * resampleTo)
        n_splits = (currentTransform.shape[0]-split_length) // split_hop
        splits = [currentTransform[i*int(split_hop):i*int(split_hop)+split_length] for i in range(n_splits)]
        filename = os.path.splitext(os.path.basename(f))[0]
        if baseroot is not None:
            suffix = os.path.dirname(f).replace(baseroot, '')
            current_path = output_path + '/' + suffix
        if not os.path.isdir(current_path):
            os.makedirs(current_path)
        current_files = []
        keep = [keep_silence(s) for s in splits]
        for i, current_chunk in enumerate(splits):
            if keep[i]:
                np.savez_compressed('%s/%s_%d.npz'%(current_path, filename, i), current_chunk)
                current_files.append('%s/%s_%d.npz'%(current_path, filename, i))
        files.append(current_files)
    return files
