# -*- coding: utf-8

import matplotlib.pyplot as pl
import numpy as np
import logging

def scatplot(scat, scattimes, audio=None, samplerate=44100, title=None, show=True, savefig=None, figsize=(10,5)):
    """Plot scattering data"""
    scatn = len(scat[0])
    dur = scattimes[scatn]-scattimes[0]
    scattimes = scattimes[:scatn]

    if len(scat) > 2:
        # draw two-sided plot
        columns = 2
    else:
        columns = 1

    if audio is not None:
        audiodur = float(len(audio))/samplerate
        plots = 3
    else:
        audiodur = dur
        plots = 2
        

    plix = 0
    pl.figure(figsize=figsize)

    if title:
        pl.title(title)
        
    # if we have audio, plot it
    if audio is not None:
        pl.subplot(plots, columns, plix*columns+1)
        pl.plot(np.linspace(0, audiodur, num=len(audio), endpoint=False), audio)
        pl.xlim(0, audiodur)
        plix += 1
        
    # plot 0th order (one-dimensional)
    pl.subplot(plots, columns, plix*columns+1)
    pl.plot(scattimes, scat[0])
    pl.xlim(0, audiodur)
    plix += 1
    
    # plot 1st order (two-dimensional)
    pl.subplot(plots, columns, plix*columns+1)
    sc = scat[1]
    scl = np.log10(sc+1.e-10)*20
    sclmax = np.max(scl)
    pl.imshow(scl.T, interpolation='nearest', aspect='auto', vmax=sclmax, vmin=sclmax-60, extent=(0,dur,0,scl.shape[-1]))
    pl.xlim(0, audiodur)
    
    if len(scat) > 2:
        # plot 2nd order (three-dimensional, split in slices)
        idxs = np.arange(5, scat[2].shape[1], 5)
        for idx in idxs:
            pl.axhline(idx, color='w', linestyle=':')
        
        sc = scat[2][:,idxs].transpose(1,0,2)
        scl = np.log10(sc+1.e-10)*20
        sclmax = np.max(scl)
        for i,sci in enumerate(scl):
            pl.subplot(len(idxs), 2, i*2+2)
            pl.imshow(sci.T, interpolation='nearest', aspect='auto', vmax=sclmax, vmin=sclmax-60, extent=(0,dur,0,scl.shape[-1]))
            pl.xlim(0, audiodur)
            
    if savefig:
        pl.savefig(savefig)
       
    if show:
        logging.info("Showing")
        pl.show()
