# -*- coding: utf-8 -*-

from . import GammatoneFilterbank
import numpy as np

def gammagram(signal, fs, winSize=1024, hopSize=512, freq_min=20.0, nb_bands=30, density=1.0, 
              bandwidth_factor=1.0, order=4,desired_delay_sec=0.02):
    filterbank = GammatoneFilterbank(fs, startband=0, 
                                     endband=nb_bands, normfreq=freq_min,
                                     density=density, bandwidth_factor = bandwidth_factor,
                                     desired_delay_sec=desired_delay_sec)
    print("[Gammatone] - filtering..")
    filters = [band[0] for band in filterbank.analyze(signal)]
    wpos = [p*hopSize for p in range(0,signal.shape[0]//hopSize)]
    gammagram = np.zeros((len(wpos), len(filters)))
    print("[Gammatone] - rendering..")
    for i in range(len(wpos)):
        for j in range(len(filters)):
            gammagram[i,j] = np.sqrt(np.mean(np.power(np.abs(filters[j][i*hopSize:(i*hopSize+winSize)]), 2)))
    return gammagram
    
    
    