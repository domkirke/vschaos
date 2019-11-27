# -*- coding: utf-8

import numpy as np
import logging
from itertools import chain, zip_longest
from .util import nextpow2, isreal
from .convolve import ConvSignal, ConvKernel


def convolve_last_dim(x, kernel, ds=1):
    """
    Perform convolution over the last dimension of x.
    Bring multi-dimensional x into a 2-dimensional form by flattening the front dimensions,
    then convolve and finally bring it back into the original dimensionality
    """ 
    xsh = x.shape
    x = x.reshape(-1,xsh[-1]) # two-dimensional
    
    maxsiglen = x.shape[1] # same as: max(len(xi) for xi in x)
    maxfltlen = max(len(ki) for ki in kernel)
    commonlen = maxsiglen
    commonpad = maxfltlen #-1

    # let's assume that maxsiglen >= maxfltlen in most use cases
    # consequently we don't have to account for different kernel sizes
    convsize = nextpow2(commonlen+commonpad*2)

    # all signals have the same size    
    convsignals = [ConvSignal(xi, pad=commonpad, convsize=convsize) for xi in x]
    # kernels may have different sizes
    convkernels = [ConvKernel(ki, n=commonlen, pad=commonpad, convsize=convsize) for ki in kernel]

    # perform the convolution
    conv = [[s.convolve(ki) for s in convsignals] for ki in convkernels]

    # downsampling after the filtering    
    total_ds = np.min(ds)
    conv = np.asarray(conv)[...,:maxsiglen:total_ds]
    
    # bring the result back into multi-dimensional form
    return conv.reshape((len(kernel),)+xsh[:-1]+(maxsiglen//total_ds,))


def scatter(audio, psi, phi, psi_ds=(None,), phi_ds=(None,)):
    """Perform scattering on input stream (sequence of audio blocks)"""
    
    siglen = nextpow2(len(audio))
    last = np.zeros(siglen, dtype=float)
    last[:len(audio)] = audio[:len(last)]
    
    scats = []
    total_ds = 1
    
    for o,(phio,psio,phio_ds,psio_ds) in enumerate(zip_longest(phi, psi, phi_ds, psi_ds, fillvalue=None)):
        logging.info("Scattering order %i" % o)
        # phi is symmetric, the time-domain kernel is real
        assert isreal(phio)

        if phio_ds is None:
            ds = 1
        else:
            ds = np.maximum([phio_ds], total_ds)//total_ds
            
        scat = convolve_last_dim(last, [phio], ds)[0].T
        assert isreal(scat)
        scats.append(scat.real)
        
        if psio is None:
            # we are done
            break

        if psio_ds is None:
            ds = 1
        else:
            ds = np.maximum(psio_ds, total_ds)//total_ds
        
        presh = last.shape[-1]
        # if psi is not hermitean, the time-domain kernel is complex
        last = convolve_last_dim(last, psio, ds)
        postsh = last.shape[-1]
        # last is complex or real
        last = np.abs(last)
        # last is real

        ds = presh//postsh
        total_ds *= ds
        
    return scats


def synthesize(frames, hopsize, wndfun=np.hanning, nframes=None):
    """Synthesize transformed frames by overlap-adding"""
    if nframes is None:
        try:
            nframes = len(frames) # number of frames
        except TypeError:
            # convert to list first
            frames = list(frames)
            nframes = len(frames)

    frames = iter(frames)
    frame0 = next(frames)
    frames = chain((frame0,), frames)
    
    fsize = frame0.shape[0] # frame size
    dims = frame0.shape[1:] # 'order' additional dimensions 
    
    if fsize%hopsize != 0:
        raise RuntimeError("Scattering frames are too short to be properly overlapped for synthesis")
    
    hopmult = fsize//hopsize
    n = (nframes+2*hopmult-1)*hopsize+fsize

    # pre-allocate result
    synth = np.zeros((n,)+dims, dtype=frame0.dtype)

    # compute synthesis window
    window = np.zeros(fsize, dtype=frame0.dtype)
    window[fsize//2-hopsize:][:hopsize*2] = wndfun(hopsize*2+1)[1:]
    
    # padding
    pad = np.zeros_like(frame0)
    
    # overlap-add with synthesis window
    for fi,f in enumerate(chain([pad]*hopmult,frames,[pad]*hopmult)):
        assert len(f) == fsize
        target = synth[fi*hopsize:fi*hopsize+fsize]
        target[:] += ((f.T*window).T)[:len(target)]
        
    return synth[hopsize*hopmult:hopsize*hopmult+nframes*fsize]
