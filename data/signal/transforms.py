# -*- coding: utf-8 -*-
"""

 Import toolbox       : Signal processing transforms

 This file contains the definition of all signal processing operations
 Currently implemented transforms are
     'raw'                # Raw waveform
     'stft'               # Short-Term Fourier Transform
     'mel'                # Log-amplitude Mel spectrogram
     'mfcc'               # Mel-Frequency Cepstral Coefficient
     'chroma'             # Chromagram
     'cqt'                # Constant-Q Transform
     'gabor'              # Gabor features
     'nsgt'               # Non-Stationary Gabor Transform
     'gammatone'          # Gammatone spectrum
     'mdct'               # Modified Discrete Cosine Transform
     'strf'               # Spectro-Temporal Receptive Fields
     'modulation'         # Modulation spectrum
     'hartley'            # Hartley transform
     'wavelet'            # Wavelet transform
     'scattering'         # Scattering transform
     'cochleogram'        # Cochleogram

 Author               : Philippe Esling
                        <esling@ircam.fr>

"""


import os, pdb
import time
import numpy as np
try:
    from skimage.transform import resize
except:
    print('Skip skimage')
import scipy.signal as scs
import scipy.io as sio
import math as mh
try:
    from matplotlib import pyplot as plt
except:
    import matplotlib 
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
# Import all relevant toolboxes
#import mdct
import pywt
import librosa
from . import rawTransform as raw

from . import nsgt
NSGT = nsgt.cq.NSGT
Scale = nsgt.fscale.Scale
OctScale = nsgt.fscale.OctScale
LogScale = nsgt.fscale.LogScale
MelScale = nsgt.fscale.MelScale


from .strf import strf, istrf
#from .scatter1d import scatter_audio, synthesize
#from .modulation import modspec, inv_modspec
#from .pyfilterbank_acids import gammagram
# from .csft import csft, icsft
from .sms_models import spsModel as SPS
    
# Quick and Dirty definition
class ErbScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0, earQ=9.26449, minBW=24.7, order=1):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        @param earQ=9.26449, minBW=24.7, order=1 - Default Glasberg and Moore parameters.
        """
        self.low = float(fmin)
        self.high = float(fmax)
        self.N = bnds
        Scale.__init__(self, bnds+beyond*2)
        self.freqs = -(earQ * minBW) + np.exp((np.arange(self.N)) * (-np.log(self.high + earQ * minBW) + np.log(self.low + earQ * minBW)) / (self.N-1)) * (self.high + earQ * minBW)
        self.freqs = self.freqs[::-1]
        self.earQ = earQ;
        self.minBW = minBW
        
    def F(self, bnd=None):
        if (np.floor(bnd) != bnd):
            return -(self.earQ * self.minBW) + np.exp((self.N - bnd) * (-np.log(self.high + self.earQ * self.minBW) + np.log(self.low + self.earQ * self.minBW)) / (self.N-1)) * (self.high + self.earQ * self.minBW)
        return self.freqs[bnd]

def transformHandler(sig, currentType, direction, options):
    # Check the input arguments
    resampleTo = int(options.get('resampleTo', 22050)) or 22050
    # Raw waveform argument
    chunkSize = int(options.get('chunkSize', 0))or 0
    chunkHop =  int(options.get('chunkHop', int(chunkSize / 2))) or int(chunkSize / 2)
    # Generic transforms arguments
    winSizeMs = int(options.get('winSizeMs', 50)) or 50
    hopSizeMs = int(options.get('hopSizeMs', winSizeMs/4)) or winSizeMs / 4
    winSize = int(options.get('winSize', int(np.power(2, np.ceil(np.log2(round((winSizeMs / 1000) * resampleTo)))))))
    hopSize = int(options.get('hopSize', int(np.power(2, np.ceil(np.log2(round((hopSizeMs / 1000) * resampleTo)))))))
    nFFT = int(options.get('nFFT', winSize)) or winSize
    # Mel, CQT and NSGT related parameters (cf. mel-gabor toolbox)
    minFreq = int(options.get('minFreq', 30)) or 30
    maxFreq = int(options.get('maxFreq', 11000)) or 11000
    nbBands = int(options.get('nbBands', 256)) or 256
    # Mfcc-related (cf. mel-gabor toolbox)
    nbCoeffs = int(options.get('nbCoeffs', 13)) or 13
    # CQT-related (cf. cqt toolbox)
    cqtBins = int(options.get('cqtBins', 360)) or 360
    cqtBinsOctave = int(options.get('cqtBinsPerOctave', 60)) or 60
    cqtGamma = int(options.get('cqtGamma', 0.5)) or 0.5
    # NSGT-related (cf. nsgt toolbox)
    nsgtBins = int(options.get('nsgtBins', 48)) or 48
    # Gammatone-related (cf. gammatone toolbox)
    gammatoneBins = int(options.get('gammatoneBins', 64)) or 64
    gammatoneMin = int(options.get('gammatoneMin', minFreq)) or minFreq
    # Chroma-related
    chromaEnhance = int(options.get('chromaEnhance', 1)) or True
    # Wavelet-related (cf. scattering toolbox)
    waveletType = options.get('waveletType', 'db2') or 'db2'
    # Scattering-related (cf. scattering toolbox)
    #scatteringDefault = options.get('scatteringDefault') or 1
    #scatteringTypes = options.get('scatteringTypes') or {'gabor_1d', 'morlet_1d', 'morlet_1d'}
    scatteringQ = options.get('scatteringQ', [8,2,1]) or [8, 2, 1]
    #scatteringT = options.get('scatteringT') or 8192
    # Cochleogram-related (cf. scattering toolbox)
    cochleogramFrame = int(options.get('cochleogramFrame', 64)) or 64        # frame length, typically, 8, 16 or 2^[natural #] ms.
    cochleogramTC = int(options.get('cochleogramTC', 16)) or 16              # time const. (4, 16, or 64 ms), if tc == 0, the leaky integration turns to short-term avg.
    cochleogramFac = int(options.get('cochleogramFac', -1)) or -1            # nonlinear factor (typically, .1 with [0 full compression] and [-1 half-wave rectifier]
    cochleogramShift = int(options.get('cochleogramShift', 0)) or 0         # shifted by # of octave, e.g., 0 for 16k, -1 for 8k,
    # Sinusoidal modeling parameters
    spsWin = options.get('spsWin', 'hann') or 'hann'
    spsWin_s = int(options.get('spsWin_s', 1023)) or 1023
    spsFft_s = int(options.get('spsFft_s', int(pow(2, np.ceil(mh.log(spsWin_s, 2))))))
    spsOverlap = int(options.get('spsOverlap', 0.6125)) or 0.6125
    spsHop_s = int(options.get('spsHop_s', int((1-spsOverlap)*spsFft_s)))
    spsT =  int(options.get('spsT', -80)) or -80                               # magnitude threshold in negative dB - def. -80
    spsMinSineDur = float(options.get('spsMinSineDur', 0.02)) or 0.02            # minimum duration of sinusoidal tracks - def. 0.02
    spsMaxnSines = int(options.get('spsMaxnSines', 150)) or 150               # maximum number of parallel sinusoids - def. 150
    spsFreqDevOffset = int(options.get('spsFreqDevOffset', 10)) or 10        # maximum frequency deviation allowed in the sinusoids from frame to frame at frequency 0 - def. 10
    spsFreqDevSlope = float(options.get('spsFreqDevSlope', 0.001)) or 0.001       # slope of the frequency deviation, higher frequencies have bigger deviation - def. 0.001
    spsStocf = float(options.get('spsStocf', 0.2)) or 0.2
    # Modulation-spectrum related (cf. modulation)
    modSize = options.get('modSize') or 'p'                         # Hop window size for modulations computation
    # STRF-related (cf. auditory toolbox)
    strfFullT = int(options.get('strfFullT', 0)) or 0                       # fullT (fullX): fullness of temporal (spectral) margin in [0, 1].
    strfFullX = int(options.get('strfFullX', 0)) or 0
    strfBP = int(options.get('strfBP', 1)) or 1                             # Pure Band-Pass indicator
    if options.get('strfRv') is not None:
        strfRv = options.get('strfRv')
    else:
        strfRv = [2.0,2.8284,4.0,5.6569,8.0,11.3137,16.0,22.6274,32.0]
    if options.get('strfRv') is not None:
        strfSv = options.get('strfSv')
    else:
        strfSv = [0.25,0.3536,0.50,0.7071,1.00,1.4142,2.0,2.8284,4.00,5.6569,8.0]
    strfMean = int(options.get('strfMean', 0)) or 0                         # Only produce the mean activations
    # CSFT-related
    csftDensity = int(options.get('csftDensity', 512)) or 512
    csftNormalize = int(options.get('csftNormalize', 1)) or True
    # Compute the forward transform
    if (direction == 'forward'):                
        if (currentType == 'stft'):
            if nFFT<winSize:
                nFFT = winSize
            currentTransform = librosa.stft(sig, n_fft=nFFT, hop_length=hopSize, win_length=winSize)
            removePhase = options.get('removePhase') or False
            if (removePhase):
                currentTransform = np.abs(currentTransform)
            # Concatenate the phase with the real part
            concatenatePhase = options.get('concatenatePhase') or False
            if (concatenatePhase):
                currentTransform = np.vstack((np.abs(currentTransform), np.angle(currentTransform)))
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'mel'):
            currentTransform = librosa.feature.melspectrogram(sig, sr=resampleTo, n_fft=nFFT, n_mels=nbBands, hop_length=hopSize, fmin=minFreq, fmax=maxFreq)
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'mfcc'):
            currentTransform = librosa.feature.mfcc(sig, sr=resampleTo, n_mfcc=nbCoeffs)
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'chroma'):
            # Implemented the librosa enhancements
            if (chromaEnhance):
                sig = librosa.effects.harmonic(y=sig, margin=8)
            currentTransform = librosa.feature.chroma_cqt(sig, sr=resampleTo, hop_length=hopSize, bins_per_octave=12*3)
            if (chromaEnhance):
                currentTransform = np.minimum(currentTransform, librosa.decompose.nn_filter(currentTransform, aggregate=np.median, metric='cosine'))
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'cqt'):
            currentTransform = librosa.cqt(sig, sr=resampleTo, hop_length=hopSize, fmin=minFreq, n_bins = cqtBins, bins_per_octave=cqtBinsOctave, filter_scale=cqtGamma)
            pdb.set_trace()
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'gabor'):
            print('[Warning] Gabor not implemented yet.')
            currentTransform = np.zeros((1, 1));
        elif (currentType == 'nsgt'):
            # Create a scale
            scl = LogScale(minFreq, maxFreq, nsgtBins * 8)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, len(sig), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.array(list(nsgt.forward(sig)))
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'nsgt-cqt'):
            # Create a scale
            scl = OctScale(minFreq, maxFreq, nsgtBins)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, len(sig), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.array(list(nsgt.forward(sig)))
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'nsgt-mel'):
            # Create a scale
            scl = MelScale(minFreq, maxFreq, nsgtBins * 8)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, len(sig), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.array(list(nsgt.forward(sig)))
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'nsgt-erb'):
            # Create a scale
            scl = ErbScale(minFreq, maxFreq, nsgtBins * 8)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, len(sig), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.array(list(nsgt.forward(sig)))
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'gammatone'):
            currentTransform = gammagram(sig, resampleTo, winSize, hopSize, nb_bands=gammatoneBins, freq_min=gammatoneMin)
        elif (currentType == 'dct'):
            currentTransform = mdct.mdct(sig, framelength=winSize, hopsize=hopSize)
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'hartley'):
            f, t, STFT = scs.stft(sig, fs=resampleTo, nfft=nFFT, noverlap=(winSize - hopSize), nperseg=winSize, return_onesided=False)
            currentTransform = np.real(STFT) - np.imag(STFT)
            currentTransform = np.transpose(currentTransform)
        elif (currentType == 'wavelet'):
            currentTransform = pywt.dwt(sig, waveletType, 'smooth')
        elif (currentType == 'scattering'):
            currentTransform, times = scatter_audio.scatter1D(sig, sr = resampleTo, M=2, Q=[8], fmin=[minFreq], fmax=[maxFreq], log=False)
            currentTransform = np.concatenate((currentTransform[1][:,:,np.newaxis], currentTransform[2]), 2)
            #currentTransform = np.array(currentTransform)
            #currentTransform = np.transpose(currentTransform)
        elif (currentType == 'sinusoidal'):
            spsW = scs.get_window(spsWin, spsWin_s, fftbins=True)
            tfreq, tmag, tphase, stocEnv = SPS.spsModelAnal(sig, resampleTo, spsW, spsFft_s, spsHop_s, spsT, spsMinSineDur, spsMaxnSines, spsFreqDevOffset, spsFreqDevSlope, spsStocf)
            currentTransform = {}
            currentTransform["frequency"] = tfreq
            currentTransform["magnitude"] = tmag
            currentTransform["phase"] = tphase
            currentTransform["stochastic"] = stocEnv
        elif (currentType == 'strf'):
            # Forward transform 
            if nFFT<winSize:
                nFFT = winSize
            # Compute first transform (here MDCT)
            currentTransform = mdct.mdct(sig, framelength=winSize, hopsize=hopSize)
            # Compute STRF transform
            currentTransform = strf(currentTransform, bp=strfBP, rates=strfRv, scales=strfSv, fl=cochleogramFrame, tc=cochleogramTC, fac=cochleogramFac, shift=cochleogramShift, full_T=strfFullT, full_X=strfFullX)
            if (strfMean):
                currentTransform = np.mean(np.mean(currentTransform[:, :, :, :], axis=0), axis=0)
        elif (currentType == 'strf-nsgt'):
            # Create a scale
            scl = ErbScale(minFreq, maxFreq, nsgtBins * 8)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, len(sig), real=True, matrixform=True, reducedform=1)
            # Forward transform 
            currentTransform = np.array(list(nsgt.forward(sig)))
            # Use log-amplitude scale
            currentTransform = np.log1p(np.abs(currentTransform))
            # Rescale the corresponding transform
            currentTransform = resize(currentTransform, (currentTransform.shape[0], int(currentTransform.shape[1] / 10)), mode='constant')
            # Compute STRF transform
            currentTransform = strf(currentTransform, bp=strfBP, rates=strfRv, scales=strfSv, tc=cochleogramTC, fac=cochleogramFac, shift=cochleogramShift, full_T=strfFullT, full_X=strfFullX)
            # Use log-amplitude scale
            currentTransform = np.log1p(np.abs(currentTransform))
            if (strfMean):
                currentTransform = np.mean(np.mean(currentTransform[:, :, :, :], axis=0), axis=0)
        elif (currentType == 'modulation'):
            currentTransform = modspec(sig, resampleTo, first='STFT')
        elif (currentType == 'modulation-mel'):
            currentTransform = modspec(sig, resampleTo, first='MEL')
        elif (currentType == 'modulation-nsgt'):
            currentTransform = modspec(sig, resampleTo, first='NSGT')
        elif (currentType == 'csft'):
            currentTransform = csft(sig, sr=resampleTo, winSize=winSize, hopSize=hopSize, density=csftDensity, normalize=csftNormalize)
        elif (currentType == 'raw'):
            currentTransform = raw.splitRawSignal(sig, {'resampleTo': resampleTo, 'chunkSize':chunkSize , 'chunkHop': chunkHop})
        else:
            raise ValueError('Unknown transform ' + currentType);
    elif (direction == 'inverse'):
        if (currentType == 'stft'):
            sig = np.transpose(sig)
            if sig.shape[1] == 1:
                sig = sig.repeat(2, -1)
            currentTransform = librosa.istft(sig.squeeze(), hop_length=hopSize, win_length=winSize)
        elif (currentType == 'mel'):
            print('[Warning] Mel inversion not implemented yet.')
            currentTransform = np.zeros((1, 1));
        elif (currentType == 'mfcc'):
            print('[Warning] MFCC is non inversible.')
            currentTransform = np.zeros((1, 1));
        elif (currentType == 'chroma'):
            print('[Warning] Chroma is non inversible.')
            currentTransform = np.zeros((1, 1));
        elif (currentType == 'cqt'):
            currentTransform = librosa.cqt(sig, sr=resampleTo, hop_length=hopSize, fmin=minFreq, bins_per_octave=cqtBins, filter_scale=cqtGamma)
        elif (currentType == 'gabor'):
            print('[Warning] Gabor inversion not implemented yet.')
            currentTransform = np.zeros((1, 1));
        elif (currentType == 'nsgt'):
            sig = np.transpose(sig)
            # Create a scale
            scl = LogScale(minFreq, maxFreq, nsgtBins * 8)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, int(options['targetDuration'] * resampleTo), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.array(list(nsgt.backward(sig)))
        elif (currentType == 'nsgt-cqt'):
            sig = np.transpose(sig)
            # Create a scale
            scl = OctScale(minFreq, maxFreq, nsgtBins)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, int(options['targetDuration'] * resampleTo), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.real(nsgt.backward(sig))
        elif (currentType == 'nsgt-mel'):
            sig = np.transpose(sig)
            # Create a scale
            scl = MelScale(minFreq, maxFreq, nsgtBins * 8)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, int(options['targetDuration'] * resampleTo), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.array(list(nsgt.backward(sig)))
        elif (currentType == 'nsgt-erb'):
            sig = np.transpose(sig)
            # Create a scale
            scl = ErbScale(minFreq, maxFreq, nsgtBins * 8)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, int(options['targetDuration'] * resampleTo), real=True, matrixform=True, reducedform=1)
            # forward transform 
            currentTransform = np.array(list(nsgt.backward(sig)))
        elif (currentType == 'gammatone'):
            print('[Warning] Gammatone is not inversible.')
            currentTransform = np.zeros((1, 1));
        elif (currentType == 'dct'):
            sig = np.transpose(sig)
            currentTransform = mdct.imdct(sig, framelength=winSize, hopsize=hopSize)
        elif (currentType == 'hartley'):
            sig = np.transpose(sig)
            iSTHTs = np.fft.fft(sig, axis=0)
            iSTHTs = (np.real(iSTHTs)-np.imag(iSTHTs)).astype('float32')
            currentTransform = np.zeros((winSize+hopSize*(iSTHTs.shape[1]-1)))
            currentTransform[0:winSize] = iSTHTs[:,0]
            for win_id in range(1,iSTHTs.shape[1]): # seems ok for any overlap>50%
                curr_start = win_id*hopSize
                currentTransform[curr_start:curr_start+nFFT] += iSTHTs[:,win_id]
        elif (currentType == 'wavelet'):
            a, b = sig
            currentTransform = pywt.idwt(a, b, 'db2', 'smooth')
        elif (currentType == 'scattering'):
            sig = np.transpose(sig)
            currentTransform = synthesize(sig, hopsize = 2200)
        elif (currentType== 'csft'):
            currentTransform = icsft(sig, sr=resampleTo)
        elif (currentType == 'strf'):
            # Compute inverse STRF transform
            currentTransform = istrf(sig, bp=0, rates=strfRv, scales=strfSv, fl=cochleogramFrame, tc=cochleogramTC, fac=cochleogramFac, shift=cochleogramShift, full_T=strfFullT, full_X=strfFullX)
            # Inverse first transform 
            currentTransform = mdct.imdct(currentTransform, framelength=winSize, hopsize=hopSize)
        elif (currentType == 'strf-nsgt'):
            # Come back from log-amplitude
            currentTransform = np.exp(sig) - 1
            # Compute inverse STRF transform
            currentTransform = istrf(currentTransform, bp=strfBP, rates=strfRv, scales=strfSv, tc=cochleogramTC, fac=cochleogramFac, shift=cochleogramShift, full_T=strfFullT, full_X=strfFullX)
            # Rescale the corresponding
            currentTransform = resize(np.abs(currentTransform), (values.shape[0], values.shape[1] * 10))
            # Create a scale
            scl = ErbScale(minFreq, maxFreq, nsgtBins * 10)
            # Calculate transform parameters
            nsgt = NSGT(scl, resampleTo, int(options['targetDuration'] * resampleTo), real=True, matrixform=True, reducedform=1)
            # Forward transform 
            currentTransform = np.array(list(nsgt.backward(currentTransform)))
            #if (strfMean):
            #    currentTransform = np.mean(np.mean(currentTransform[:, :, :, :], axis=0), axis=0)
        elif (currentType == 'modulation'):
            currentTransform = inv_modspec(sig, resampleTo, first='STFT')            
        elif (currentType == 'sinusoidal'):
            sig = sig.item()
            y, ys, yst = SPS.spsModelSynth(sig["frequency"], sig["magnitude"], sig["phase"], sig["stochastic"], spsFft_s, spsHop_s, resampleTo)
            currentTransform = y
        elif (currentType == 'descriptors'):
            print('[Warning] Descriptors not implemented yet.')
            currentTransform = np.zeros((1, 1));
        elif(currentType=='raw'):
            currentTransform=raw.concatSignal(sig, window_size=chunkSize)
        else:
            raise ValueError('Unknown transform ' + currentType);
    return currentTransform

def computeTransform(audioList, transformType, options={}, out=None):
    # Check the input arguments
#    options = options['transformParameters']
    resampleTo = options.get('resampleTo') or 22050
    
    # Normalization parameters
    targetDuration = int(options.get('targetDuration', 0)) or 0
    normalizeInput = int(options.get('normalizeInput', 0)) or False
    normalizeOutput = int(options.get('normalizeOutput', 0)) or False
    equalizeHistogram = int(options.get('equalizeHistogram', 0)) or False
    logAmplitude = int(options.get('logAmplitude', 0)) or False
    downsampleFactor = int(options.get('downsampleFactor', 0)) or 0
    
    # Debug mode
    debugMode = options.get('debugMode') or False
    testTime = options.get('testTime') or False
    
    # Create the analysis folder
    # If the transform type "all" is asked, fill transform with all known ones
    if ('all' in transformType):
        transformType = ['stft', 'mel', 'mfcc', 'chroma', 'cqt', 'nsgt', 'nsgt-cqt', 'nsgt-mel', 'nsgt-erb', 'gammatone', 'dct', 'hartley', 'wavelet', 'scattering', 'sinusoidal', 'strf', 'strf-nsgt', 'modulation', 'modulation-mel', 'modulation-nsgt', 'csft', 'raw']
    # Final result
    
    resultList = [None] * len(audioList)
    # Parse through the mini-batch
    for i in range(len(audioList)):
        # Current audio file
        audioFile = audioList[i];
        if audioFile is None:
            continue
        
        # Read the corresponding signal
        print('     - Processing ' + audioFile)
        current_ext = os.path.splitext(audioFile)[1]
        if current_ext == ".npz":
            with np.load(audioFile) as f:
                sig = f['audio']
                fs = 22050
        else:
            sig, fs = librosa.load(audioFile)
        # Turn to mono if multi-channel file
        if (len(sig.shape) > 1 and sig.shape[1] > 1):
            sig = np.mean(sig, 2)
            
        # First resample the signal (similar to ircamDescriptor)
        if (fs != resampleTo):
            sig = scs.resample(sig, int(resampleTo * (len(sig) / float(fs))))
            fs = resampleTo
            
        # Now ensure that we have the target duration
        if (targetDuration):
            # Input is longer than required duration
            if ((len(sig) / fs) > targetDuration):
                # Select the center of the sound and trim around
                #midPoint = np.floor(len(sig) / 2); 
                #midLen = np.floor((targetDuration * fs) / 2);
                #sPoint = midPoint - midLen; 
                #ePoint = midPoint + midLen;
                #sig = sig[int(sPoint):int(ePoint)];
                sig = sig[:int((targetDuration * fs))]
            # Otherwise pad with zeros
            else:
                #sig = np.padarray(sig, np.floor(((targetDuration * fs) - len(sig)) / 2) + 1);
                sig = np.pad(sig, int(np.round(targetDuration * fs)) - len(sig), 'constant', constant_values = 0);
                
        # Check if we need to normalize the input
        if (normalizeInput):
            sig = sig / np.max(sig);
            
        # Compute all transforms
        if (testTime):
            startTime = time.perf_counter();
        # Call the appropriate handler
        currentTransform = transformHandler(sig, transformType, 'forward', options)
        
        # Put in log-amplitude scale
        if (logAmplitude):
            #if not removePhase or concatenatePhase:
            #    currentTransform[0:int(options['nFFT']/2+1), :] = np.log1p(currentTransform[0:int(options['nFFT']/2+1), :])
            #else:
                currentTransform = np.log1p(np.abs(currentTransform))
                
        if (downsampleFactor > 0):
            # Rescale the corresponding transform
            if (not np.iscomplexobj(currentTransform)):
                currentTransform = resize(currentTransform, (int(currentTransform.shape[0] / downsampleFactor), currentTransform.shape[1]), mode='constant')
            else:
                currentTransform_abs = np.abs(currentTransform)
                currentTransform_abs = resize(currentTransform_abs, (int(currentTransform_abs.shape[0] / downsampleFactor), currentTransform_abs.shape[1]), mode='constant')
                currentTransform_phase = np.angle(currentTransform)
                currentTransform_phase = resize(currentTransform_phase, (int(currentTransform_phase.shape[0] / downsampleFactor), currentTransform_phase.shape[1]), mode='constant')
                currentTransform = currentTransform_abs * np.exp(currentTransform_phase*1j)
                
       # Normalize output
        if (normalizeOutput):
            currentTransform = (currentTransform - np.mean(currentTransform)) / np.max(np.abs(currentTransform))
        # Equalize histogram
        if (equalizeHistogram):
            print('[Warning] Feature not implemented in Python.')
            
        # Plot the current transform if debugging
        if (debugMode):
            if (len(currentTransform.shape) == 2):
                plt.figure(figsize=(14,6)) 
                plt.imshow(np.flipud(np.abs(currentTransform)))
                plt.title(transformType)
                plt.axis('tight')
                #librosa.display.specshow(currentTransform, sr=resampleTo, x_axis='time', y_axis='linear')
            else:
                plt.figure(figsize=(18,6)) 
                plt.imshow(np.mean(np.mean(np.abs(currentTransform), axis=0), axis=0))
                plt.title(transformType)
                plt.axis('tight')
        if (testTime):
            print('Time to compute %s \t : %f'%(transformType, time.perf_counter() - startTime));
            
        # add transform to list
        resultList[i] = currentTransform

        if out is not None:
            dirname = '/'.join(os.path.abspath(out).split('/')[:-1])
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            save_memmap(currentTransform, out)
            # if issubclass(type(currentTransform), list):
            #     [np.savez_compressed('%s_%s'%(out, i), currentTransform[i]) for i in range(len(currentTransform))]
            # else:
            #     print(out);
            #     np.savez_compressed(out, currentTransform)
    return {'shape':currentTransform.shape, 'strides':currentTransform.strides, 'dtype':currentTransform.dtype}


def save_memmap(arr, path_out):
    if issubclass(type(arr), list):
        [save_memmap(arr[i], path_out+'_'+i) for i in range(len(arr))]
    else:
        current_memmap = np.memmap(f'{path_out}.dat', arr.dtype, 'w+', shape=arr.shape)
        current_memmap[:] = arr[:]
        del current_memmap

        

def inverseTransform(transform, transformType, options, iterations=5, output=None, method='griffin-lim', originalPhase=None, *args, **kwargs):
    options = options['transformParameters']
    # Check the input arguments
    resampleTo = options.get('resampleTo') or 22050
    # Normalization parameters
    logAmplitude = options.get('logAmplitude') or False
    # Phase-related parameters
    concatenatePhase = options.get('concatenatePhase') or False
    # Debug mode
    testTime = options.get('testTime') or False
    if (concatenatePhase):
        transform = transform[:(transform.shape[0]/2)] * np.exp(1j*transform[(transform.shape[0]/2):])
    if (logAmplitude):
        transform = np.exp(transform - 1)
    if (testTime):
        startTime = time.perf_counter();
    # Set of reverse transforms
    if (method == 'griffin-lim'):
        if transformType != 'raw':
            p = 2 * np.pi * np.random.random_sample(transform.shape) - np.pi
            for i in range(iterations):
                S = transform * np.exp(1j*p)
                inv_p = transformHandler(S, transformType, 'inverse', options)
                new_p = transformHandler(inv_p, transformType, 'forward', options)
                new_p = np.angle(new_p)
                # Momentum-modified Griffin-Lim
                p = new_p + (0.99 * (new_p - p))
    elif (method == 'direct'):
        inv_p = transformHandler(transform, transformType, 'inverse', options)
    elif (method == 'originalPhase'):
        transform = transform * np.exp(originalPhase*1j)
        inv_p = transformHandler(transform, transformType, 'inverse', options)
    if (output):
        librosa.output.write_wav(output, np.real(inv_p), resampleTo)
    if (testTime):
        print('Time to invert %s \t : %f'%(transformType, time.perf_counter() - startTime));
    return inv_p


def extractReference(audioFile, resampleTo, targetDuration):
    sig, fs = librosa.load(audioFile)
    # Turn to mono if multi-channel file
    if (len(sig.shape) > 1 and sig.shape[1] > 1):
        sig = np.mean(sig, 2)
    # First resample the signal (similar to ircamDescriptor)
    if (fs != resampleTo):
        sig = scs.resample(sig, int(resampleTo * (len(sig) / fs)))
        fs = resampleTo
    # Now ensure that we have the target duration
    if (targetDuration):
        # Input is longer than required duration
        if ((len(sig) / fs) > targetDuration):
            # Select the center of the sound and trim around
            #midPoint = np.floor(len(sig) / 2); 
            #midLen = np.floor((targetDuration * fs) / 2);
            #sPoint = midPoint - midLen; 
            #ePoint = midPoint + midLen;
            #sig = sig[int(sPoint):int(ePoint)];
            sig = sig[:np.floor((targetDuration * fs))]
        # Otherwise pad with zeros
        else:
            #sig = np.padarray(sig, np.floor(((targetDuration * fs) - len(sig)) / 2) + 1);
            sig = np.padarray(sig, (0, np.floor((targetDuration * fs) - len(sig))))# / 2) + 1);
    return sig


def reconstructionError(sig, origSpec, transformType, options):
    newSpec = transformHandler(sig, transformType, 'forward', options['transformParameters']);
    # Compute MSE reconstruction criterion
    minLen = min(newSpec.shape[0], origSpec.shape[0])
    diff = np.abs(newSpec[:minLen]) - np.abs(origSpec[:minLen])
    val = np.sum(np.sum(np.power(diff, 2)))
    return val / np.sum(np.sum(np.power(np.abs(origSpec[:minLen]), 2)))

if __name__ == '__main__':
    oldRoot = '/Users/esling/Coding/aciditools/data/sets/signal'
    newRoot = '/Users/esling/Coding/aciditools/data/sets/signal/test'
    audioList = ['test_blues.au', 'test_classical.au', 'test_drums.mp3', 'test_metal.au', 'test_nature.wav', 'test_voice.wav'];
    # List all transforms
    allTransforms = ['raw', 'stft', 'mel', 'mfcc', 'chroma', 'cqt', 'nsgt', 'nsgt-cqt', 'nsgt-mel', 'nsgt-erb', 'gammatone', 'dct', 'hartley', 'wavelet', 'scattering', 'sinusoidal', 'strf', 'strf-nsgt', 'modulation', 'modulation-mel', 'modulation-nsgt', 'csft']
    # List of all invertible transforms
    invertibleTransforms = ['raw', 'stft', 'nsgt', 'nsgt-cqt', 'nsgt-mel', 'nsgt-erb', 'dct', 'hartley', 'wavelet', 'strf', 'modulation', 'sinusoidal']
    # List of all invertible transforms
    invertiblePhaseTransforms = ['stft', 'nsgt', 'nsgt-cqt', 'nsgt-mel', 'nsgt-erb', 'strf', 'modulation']
    # Change this to test specific
    testFull = True;
    if (testFull):
        transformTypes = allTransforms
        durationTests = [0.25, 0.5, 0.75, 1, 2.1, 3, 4.5, 6, 7]
        griffinTests = [2, 5, 10, 15]
        resampleTests = [2, 5, 10]
    else:
        transformTypes = ['nsgt', 'nsgt-cqt', 'nsgt-erb']
        durationTests = np.concatenate((np.linspace(0.1, 1.1, 11)[:-1], np.linspace(1.5, 10, 21)))
        audioList = ['test_blues.au']
        griffinTests = [10]
        resampleTests = [1]
    # Create indices for tested transforms
    invertibleIDs = []
    invertiblePhaseIDs = []
    invertibleNames = []
    invertiblePhaseNames = []
    for i in range(len(transformTypes)):
        if (transformTypes[i] in invertibleTransforms):
            invertibleIDs.append(i)
            invertibleNames.append(transformTypes[i])
        if (transformTypes[i] in invertiblePhaseTransforms):
            invertiblePhaseIDs.append(i)
            invertiblePhaseNames.append(transformTypes[i])
    # Analysis options
    options = {}
    options['debugMode'] = False
    options['forceRecompute'] = True
    options['normalizeInput'] = False
    options['normalizeOutput'] = False
    options['removePhase'] = False
    options['logAmplitude'] = False 
    options['testTime'] = False
    options['targetDuration'] = 3
    finalOptions = {}
    finalOptions['transformParameters'] = options
    # Number of tests
    nbTransforms = len(transformTypes)
    nbDurations = len(durationTests)
    nbFiles = len(audioList)
    nbGriffin = len(griffinTests)
    nbResample = len(resampleTests)
    # Record all types of times
    timeForward = np.zeros((nbTransforms, nbDurations, nbFiles))
    timeBackward = np.zeros((nbTransforms, nbDurations, nbFiles))
    timeBackwardGriffin = np.zeros((nbTransforms, nbDurations, nbFiles))
    # Size of transforms
    nbDimensions = np.zeros((nbTransforms, nbDurations, nbFiles))
    # Reconstruction errors
    recErrorsDirect = np.zeros((nbTransforms, nbDurations, nbFiles))
    recErrorsGriffin = np.zeros((nbTransforms, nbDurations, nbFiles))
    # Griffin-lim specific analysis
    griffinError = np.zeros((nbTransforms, nbGriffin, nbFiles))
    griffinTime = np.zeros((nbTransforms, nbGriffin, nbFiles))
    # Resampling analysis
    resampleError = np.zeros((nbTransforms, nbResample, nbFiles))
    resampleTime = np.zeros((nbTransforms, nbResample, nbFiles))
    # Check their respective sizes
    for tI in range(nbTransforms):
        t = transformTypes[tI]
        print('- Testing ' + t)
        for dI in range(nbDurations):
            d = durationTests[dI]
            options['targetDuration'] = d
            print('   . Duration = ' + str(d) + ' s')
            for fI in range(nbFiles):
                #try:
                    f = audioList[fI]
                    # Extract the reference signal
                    refSignal = extractReference(f, 22050, d)
                    # Compute transforms
                    startTime = time.perf_counter();
                    res = computeTransform([f], [t], '', 'test', finalOptions)
                    timeForward[tI][dI][fI] = time.perf_counter() - startTime
                    print('        * Forward time \t : ' + str(timeForward[tI][dI][fI]))
                    # Import corresponding transform
                    values = np.load(res[0])
                    print('        * Size forward \t : ' + str(values.shape))
                    nbDimensions[tI][dI][fI] = values.size
                    if (t in invertibleTransforms):
                        startTime = time.perf_counter();
                        sig = inverseTransform(values, t, finalOptions, iterations = 0, output='results/inv_'+f+'_direct_'+t+'.wav', method='direct')
                        timeBackward[tI][dI][fI] = time.perf_counter() - startTime
                        print('        * Direct inv. \t : ' + str(timeBackward[tI][dI][fI]))
                        if (t in ['wavelet', 'sinusoidal']):
                            continue
                        recErrorsDirect[tI][dI][fI] = reconstructionError(sig, values, t, finalOptions)
                        print('        * Direct err. \t : ' + str(recErrorsDirect[tI][dI][fI]))
                        if (t in ['dct', 'hartley', 'sinusoidal', 'raw']):
                            continue
                        startTime = time.perf_counter();
                        sig = inverseTransform(np.abs(values), t, finalOptions, iterations = 3, output='results/inv_'+f+'_griffin_'+t+'.wav', method='griffin-lim')
                        timeBackwardGriffin[tI][dI][fI] = time.perf_counter() - startTime
                        print('        * Griffin inv. \t : ' + str(timeBackwardGriffin[tI][dI][fI]))
                        # Compute MSE reconstruction criterion
                        minLen = min(len(refSignal), len(sig))
                        recErrorsGriffin[tI][dI][fI] = reconstructionError(sig, values, t, finalOptions)
                        print('        * Griffin err. \t : ' + str(recErrorsGriffin[tI][dI][fI]))
                        # Perform specfic tests only for the 2 second sample
                        if (d == 2.1):
                            print('        [Specific Griffin-Lim tests]')
                            # Griffin-Lim tests
                            for gI in range(nbGriffin):
                                g = griffinTests[gI]
                                startTime = time.perf_counter();
                                sig = inverseTransform(np.abs(values), t, finalOptions, iterations=g, output='results/inv_'+f+'_griffin_'+t+'_it'+str(g)+'.wav', method='griffin-lim')
                                griffinTime[tI][gI][fI] = time.perf_counter() - startTime
                                print('           . ' + str(g) + ' iters inv. \t : ' + str(griffinTime[tI][gI][fI]))
                                # Compute MSE reconstruction criterion
                                minLen = min(len(refSignal), len(sig))
                                griffinError[tI][gI][fI] = reconstructionError(sig, values, t, finalOptions)
                                print('           . ' + str(g) + ' iters err. \t : ' + str(griffinError[tI][gI][fI]))
                            print('        [Specific resampling tests]')
                            if (t in ['strf', 'strf-nsgt']):
                                continue;
                            for rI in range(nbResample):
                                r = resampleTests[rI]
                                origTime = values.shape[0]
                                maxVal = np.max(np.abs(values))
                                newValues = resize(np.abs(values) / maxVal, (int(values.shape[0] / r), values.shape[1]))
                                newValues = resize(newValues, (int(origTime), values.shape[1])) * maxVal
                                startTime = time.perf_counter();
                                sig = inverseTransform(newValues, t, finalOptions, iterations=20, output='results/inv_'+f+'_griffin_'+t+'_down'+str(r)+'.wav', method='griffin-lim')
                                resampleTime[tI][rI][fI] = time.perf_counter() - startTime
                                print('           . ' + str(r) + ' down inv. \t : ' + str(resampleTime[tI][rI][fI]))
                                # Compute MSE reconstruction criterion
                                minLen = min(len(refSignal), len(sig))
                                resampleError[tI][rI][fI] = reconstructionError(sig, values, t, finalOptions)
                                print('           . ' + str(r) + ' down err. \t : ' + str(resampleError[tI][rI][fI]))
                #except Exception as e:
                #    print('Exception')
                #    print(e)
        # Full table to save (in case of crash)
        fullAnalysis = {'timeForward':timeForward, 'timeBackward':timeBackward,
                    'timeBackwardGriffin':timeBackwardGriffin, 'nbDimensions':nbDimensions,
                    'recErrorsDirect':recErrorsDirect, 'recErrorsGriffin':recErrorsGriffin,
                    'griffinError':griffinError, 'griffinTime':griffinTime,
                    'resampleError':resampleError, 'resampleTime':resampleTime}
        np.save('results_full.npy', fullAnalysis)
    #%%
    # Plotting the results
    #
    # List all transforms
    allTransforms = ['raw', 'stft', 'mel', 'mfcc', 'chroma', 'cqt', 'nsgt', 'nsgt-cqt', 'nsgt-mel', 'nsgt-erb', 'gammatone', 'dct', 'hartley', 'wavelet', 'scattering', 'sinusoidal', 'strf', 'strf-nsgt', 'modulation', 'modulation-mel', 'modulation-nsgt', 'csft']
    # List of all invertible transforms
    invertibleTransforms = ['raw', 'stft', 'nsgt', 'nsgt-cqt', 'nsgt-mel', 'nsgt-erb', 'dct', 'hartley', 'wavelet', 'strf', 'modulation', 'sinusoidal']
    # List of all invertible transforms
    invertiblePhaseTransforms = ['stft', 'nsgt', 'nsgt-cqt', 'nsgt-mel', 'nsgt-erb', 'strf', 'modulation']
    nbTransforms = len(allTransforms)
    # Load all results
    fullAnalysis = np.load('results_full.npy').item()
    timeForward = fullAnalysis['timeForward']
    timeBackward = fullAnalysis['timeBackward']
    timeBackwardGriffin = fullAnalysis['timeBackwardGriffin']
    nbDimensions = fullAnalysis['nbDimensions']
    recErrorsDirect = fullAnalysis['recErrorsDirect']
    recErrorsGriffin = fullAnalysis['recErrorsGriffin']
    griffinError = fullAnalysis['griffinError']
    griffinTime = fullAnalysis['griffinTime']
    resampleError = fullAnalysis['resampleError']
    resampleTime = fullAnalysis['resampleTime']
    # Forward compute times
    plt.figure()
    for t in range(nbTransforms):
        plt.plot(durationTests, np.mean(timeForward[t], axis=1))
    plt.legend(transformTypes, loc='upper left')
    plt.title('Forward computation / durations')
    plt.show()
    # Backward compute times
    plt.figure()
    for t in invertibleIDs:
        plt.plot(durationTests, np.mean(timeBackward[t], axis=1))
    plt.legend(invertibleNames, loc='upper left')
    plt.show()
    plt.title('Backward computation / durations')
    # Backward compute times
    plt.figure()
    for t in invertiblePhaseIDs:
        plt.plot(durationTests, np.mean(timeBackwardGriffin[t], axis=1))
    plt.legend(invertiblePhaseNames, loc='upper left')
    plt.show()
    plt.title('Griffin-Lim backward / durations')
    # Dimension analysis
    plt.figure()
    for t in range(nbTransforms):
        plt.plot(durationTests, np.mean(nbDimensions[t], axis=1))
    plt.legend(transformTypes, loc='upper left')
    plt.show()
    plt.title('Number of dimensions')
    # Now create overall matrices to plot boxplots
    meanForward = np.reshape(timeForward, (nbTransforms, nbDurations * nbFiles))
    plt.figure()
    plt.boxplot(meanForward.T)
    plt.xticks(range(1, nbTransforms+1), transformTypes)
    plt.title('Forward computations')
    plt.show()
    # Now create overall matrices to plot boxplots
    meanBackward = np.reshape(timeBackward[np.array(invertibleIDs)], (len(invertibleIDs), nbDurations * nbFiles))
    plt.figure()
    plt.boxplot(meanBackward.T)
    plt.xticks(range(1, len(invertibleIDs)+1), invertibleNames)
    plt.title('Backward computations')
    plt.show()
    # Now create overall matrices to plot boxplots
    meanGriffin = np.reshape(timeBackwardGriffin[np.array(invertiblePhaseIDs)], (len(invertiblePhaseIDs), nbDurations * nbFiles))
    plt.figure()
    plt.boxplot(meanGriffin.T)
    plt.xticks(range(1, len(invertiblePhaseIDs)+1), invertiblePhaseNames)
    plt.title('Griffin-Lim computations')
    plt.show()
    #
    # REAL-TIME ANALYSIS
    #
    realTimeInversible = invertibleNames.copy();
    realTimeInversible.insert(0, 'Real-time');
    realTimeDirect = timeForward + timeBackward
    plt.figure()
    # Start by plotting the real-time line
    plt.plot(durationTests, durationTests, linestyle='--')
    for t in invertibleIDs:
        plt.errorbar(durationTests, np.mean(realTimeDirect[t], axis=1), yerr=np.var(realTimeDirect[t], axis=1))
    plt.legend(realTimeInversible, loc='upper left')
    plt.title('Forward + Backward')
    plt.show()
    realTimeGriffin = timeForward + timeBackwardGriffin
    realTimePhaseInversible = invertiblePhaseNames.copy();
    realTimePhaseInversible.insert(0, 'Real-time');
    plt.figure()
    # Start by plotting the real-time line
    plt.plot(durationTests, durationTests, linestyle='--')
    for t in invertiblePhaseIDs:
        plt.errorbar(durationTests, np.mean(realTimeGriffin[t], axis=1), yerr=np.var(realTimeDirect[t], axis=1))
    plt.legend(realTimePhaseInversible, loc='upper left')
    plt.title('Forward + Griffin')
    plt.show()
    #
    # RECONSTRUCTION ERROR ANALYSIS
    #
    # Reconstruction (direct)
    plt.figure()
    for t in invertibleIDs:
        plt.plot(durationTests, np.mean(recErrorsDirect[t], axis=1))
    plt.legend(invertiblePhaseNames, loc='upper left')
    plt.title('Reconstruction error (direct)')
    plt.show()
    # Reconstruction (griffin)
    plt.figure()
    for t in invertiblePhaseIDs:
        plt.plot(durationTests, np.mean(recErrorsGriffin[t], axis=1))
    plt.legend(invertiblePhaseNames, loc='upper left')
    plt.title('Reconstruction error (Griffin-Lim - 20 iterations)')
    plt.show()
    #
    # GRIFFIN-LIM ANALYSIS
    #
    # Forward compute times
    plt.figure()
    for t in invertiblePhaseIDs:
        plt.plot(griffinTests, np.mean(griffinTime[t], axis=1))
    plt.legend(invertiblePhaseNames, loc='upper left')
    plt.title('Duration Griffin-Lim iterations')
    plt.show()
    # Reconstruction compute times
    plt.figure()
    for t in invertiblePhaseIDs:
        plt.plot(griffinTests, np.mean(griffinError[t], axis=1))
    plt.legend(invertiblePhaseNames, loc='upper left')
    plt.title('Error Griffin-Lim iterations')
    plt.show()
    #
    # RESAMPLE ANALYSIS
    #
    # Forward compute times
    plt.figure()
    for t in invertiblePhaseIDs:
        plt.plot(resampleTests, np.mean(resampleTime[t], axis=1))
    plt.legend(invertiblePhaseNames, loc='upper left')
    plt.title('Inverse time vs. resampling')
    plt.show()
    # Reconstruction compute times
    plt.figure()
    for t in invertiblePhaseIDs:
        plt.plot(resampleTests, np.mean(resampleError[t], axis=1))
    plt.legend(invertiblePhaseNames, loc='upper left')
    plt.title('Inverse error vs. resampling')
    plt.show()
    iurrent_transform = ct
