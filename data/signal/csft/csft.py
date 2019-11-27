# -*- coding: utf-8 -*-
from .DSPfunc import *
import numpy as np

__version__ = "0.01"
__author__  = "G.Presti"

def ditherIndex(n, ditherAmount, smin=0):
    dither = ditherAmount * ((np.random.normal(1,1,n)) / n);
    sampler = np.linspace(smin,1,n) + dither;
    sampler = np.maximum(np.minimum(sampler,1.0),smin);
    return sampler

def magnitude2pdf(targetPDF, x, n=None, jitter=0):
    '''Encodes a function targetPDF(x) as density of a sequence of numbers'''
    if n is None:
        n = targetPDF.size
    if np.any(targetPDF<0):
        print ("WARNING!   Target PDF contains negative values!")
    scale = np.sum(targetPDF)
    targetPDF = targetPDF/scale
    cdf = np.cumsum(targetPDF)
    sampler = ditherIndex( n, jitter, np.amin(cdf) )
    cdf[-1] = cdf[-1] + np.finfo(np.float32).eps #Stupid Python...
    worker = sp.interpolate.interp1d(cdf,x)
    randomVar = worker(sampler)    
    return randomVar, scale

def pdf2magnitude(f, D, scale):
    Mp = np.histogram(D,np.append([-1],f))[0]
    return scale * Mp / np.sum(Mp)


def csft(sig, sr, winSize=1024, hopSize=512, density=1024, normalize=True, concatenatePhase=False):
    '''Set some parameters'''
    win  = np.hanning(winSize)
    l = sig.size
    
    '''Split input into chunks (columns) and window them'''
    buf = buffer(sig,winSize,hopSize)
    buf = buf * win[:,None]
    
    '''STFT (Matlab like: rows=frequency bins; columns=time frames)'''
    X, F, T = stft(buf,hopSize,sr)
    numBins, numFrames = X.shape
    
    '''Magnitude is what we're going to encode'''
    M = np.abs(X)
    phase = np.angle(X)
    
    '''Make room for model and scale variables'''
    models = np.empty([numFrames, density])
    scales = np.empty(numFrames)
    
    '''Encode M as the density function of a regular grid + a scale factor'''
    for n in range(numFrames):
        models[n], scales[n] =  magnitude2pdf(M[:,n],F,density)
        
    '''Normalize data'''
    if normalize:
        models /= np.max(models)
        
    '''Concatenate phase'''
    transform = np.concatenate((np.expand_dims(scales, 1), models), axis=1)
    transform = np.concatenate((transform, np.transpose(phase)), axis=1)
        
    return transform
 
    
def icsft(transform, sr):
    numFrames, numBins = transform.shape
    nBins -= 1
    Mp = np.empty([numFrames, numBins])
    scales = transform[:, -1]
    
    '''Decode Mp as the density function of a random variable + a scale factor'''
    for n in range(numFrames):
        Mp[n] = pdf2magnitude(F,models[n],scales[n])

    '''My STFT and ISTFT routines are Matlab-like, so I have to swap rows and columns '''
    return Mp
