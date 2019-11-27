'''
Created on 19.01.2016

@author: thomas
'''

import numpy as np
from scipy.interpolate import interp1d


def nextpow2(n):
    return 2**int(np.ceil(np.log2(n)))


def resample(sig, size, kind='linear'):
    """Signal resampling"""
    siglen = len(sig)
    if size == siglen:
        resampled = sig
    elif size < siglen:
        assert siglen%size == 0
        resampled = sig[::siglen//size]
    else:
#        assert size%siglen == 0
        intx = np.linspace(0, siglen, num=size, endpoint=False)
        if kind == 'linear':
            # use numpy interpolation with wrap-around
            sigx = np.arange(siglen)
            resampled = np.interp(intx, sigx, sig, period=siglen)
        else:
            hsiglen = siglen//2
            sig = np.concatenate((sig[-hsiglen:],sig,sig[:hsiglen]))
            sigx = np.arange(-hsiglen, siglen+hsiglen)
            resampled = interp1d(sigx, sig, kind=kind, assume_sorted=True, copy=False)(intx)
    return resampled


def isreal(x, tol=1.e-6):
    return np.alltrue(np.abs(x.imag) <= tol)


def issymm(x, tol=1.e-6):
    """Check if x is fft-antisymmetric, so that the inverse fft be real""" 
    return np.allclose(x[1:len(x)//2], np.conj(x[len(x)-1:len(x)//2:-1]), rtol=0., atol=tol) and np.isreal(x[0]) and np.isreal(x[len(x)//2])


def pooling(x, factor, axis=0, poolfun=np.mean):
    x = np.asarray(x)
    sh = x.shape
    resdim = (sh[axis]//factor)*factor
    newsz = (slice(None),)*axis+(slice(0,resdim),)
    xtrim = x[newsz]
    reshaped = xtrim.reshape(sh[:axis]+(-1,factor)+sh[axis+1:])
    x = poolfun(reshaped, axis=axis+1)
    return x


