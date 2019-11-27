# -*- coding: utf-8 -*-
import mdct
import librosa
import numpy as np
from nsgt.cq import NSGT
from nsgt.fscale import MelScale, OctScale
from skimage.transform import resize
from matplotlib import pyplot as plt

def modspec(signal, sr, win_len=0.08, first='MEL', log=True):
    """Transform spectrogram into a modulation spectrogram"""
    if (first == 'STFT'):
        S = librosa.stft(signal, win_length=int(0.04643 * sr), hop_length=int(0.01161 * sr), window='hamming')
        S = np.abs(S)
    elif (first == 'DCT'):
        S = mdct.mdct(signal)
    elif (first == 'MEL'):
        S = librosa.feature.melspectrogram(signal, sr=sr)
        S = np.abs(S)
    elif (first == 'NSGT'):
        # Create a scale
        scl = MelScale(80, 10000, 24*8)
        # Calculate transform parameters
        nsgt = NSGT(scl, sr, len(signal), real=False, matrixform=True, reducedform=1)
        # Forward transform 
        S = np.array(list(nsgt.forward(signal)))
        S = np.log(np.abs(S))
        #S = np.log(np.abs(S))#resize(np.abs(S), (int(S.shape[0] / 10), S.shape[1]))
        df = np.linspace(80, 10000, S.shape[0])
        dt = np.linspace(0, len(signal) / sr, S.shape[1])
        wf, wt, mpsVals = mps(S, df, dt, window=0.8)
    if (first != 'NSGT'):
        # 2D FT of specgram to perform rate/scale filter in secgram-freq domain
        mpsVals = np.fft.fftshift(np.fft.fft2(S))#,s=(N2,M2))[:,0:int(M2/2)]
    if (log == True):
        mpsVals = np.log(np.abs(mpsVals))
    return mpsVals


def inv_modspec(mps, sr, first='MEL', log=True):
    """Invert an MPS back to a spectrogram.
    Parameters
    ----------
    mps : array
        The MPS to be inverted with a 2D IFFT
    out_shape : array
        The dimensions of the output spectrogram
    Returns
    -------
    ispec : array, shape == out_shape
        The inverted spectrogram
    """
    if (log == True):
        mps = np.exp(mps)
    mps = np.fft.ifftshift(mps)
    spec = np.real(np.fft.irfft2(mps))
    if (first == 'STFT'):
        S = librosa.istft(spec, win_length=int(0.04643 * sr), hop_length=int(0.01161 * sr), window='hamming')
    elif (first == 'DCT'):
        S = mdct.imdct(spec)
    elif (first == 'NSGT'):
        # Create a scale
        scl = MelScale(80, 10000, 24*8)
        # Calculate transform parameters
        nsgt = NSGT(scl, sr, len(signal), real=False, matrixform=True, reducedform=1)
        # Forward transform 
        S = np.array(list(nsgt.backward(signal)))
        S = np.log(np.abs(S))
        #S = np.log(np.abs(S))#resize(np.abs(S), (int(S.shape[0] / 10), S.shape[1]))
        df = np.linspace(80, 10000, S.shape[0])
        dt = np.linspace(0, len(signal) / sr, S.shape[1])
        wf, wt, mpsVals = mps(S, df, dt, window=0.8)
    return S

def gaussian_window(N, nstd):
    """
        Generate a Gaussian window of length N and standard deviation nstd.
    """
    hnwinlen = (N + (1-N%2)) / 2
    gauss_t = np.arange(-hnwinlen, hnwinlen+1, 1.0)
    gauss_std = float(N) / float(nstd)
    gauss_window = np.exp(-gauss_t**2 / (2.0*gauss_std**2)) / (gauss_std*np.sqrt(2*np.pi))
    return gauss_t,gauss_window

def mtfft(spectrogram, df, dt, Norm=False, Log=False):
    """
        Compute the 2d modulation transfer function for a given time frequency slice.
        return temporal_freq,spectral_freq,mps_pow,mps_phase
    """
    #normalize and mean center the spectrogram 
    sdata = spectrogram.copy()
    if Norm:
        sdata /= sdata.max()
        sdata -= sdata.mean()
    #take the 2D FFT and center it
    smps = np.fft.fft2(sdata)
    smps = np.fft.fftshift(smps)
    #compute the log amplitude
    mps_pow = np.abs(smps)**2
    if Log:
        mps_pow = 10*np.log10(mps_pow)
    #compute the phase
    mps_phase = np.angle(smps)
    #compute the axes
    nf = mps_pow.shape[0]
    nt = mps_pow.shape[1]
    spectral_freq = np.fft.fftshift(np.fft.fftfreq(nf, d=df[1]-df[0]))
    temporal_freq = np.fft.fftshift(np.fft.fftfreq(nt, d=dt[1]-dt[0]))
    return spectral_freq, temporal_freq, mps_pow, mps_phase

def mps(spectrogram, df, dt, window=None, Norm=False):
    """
    Calculates the modulation power spectrum using overlap and add method 
    with a gaussian window of length window in s
    """
    # Check the size of the spectrogram vs dt
    nt = dt.size
    nf = df.size
    if spectrogram.shape[1] != nt and spectrogram.shape[0] != nf:   
        print('Error in mps. Expected  %d bands in frequency and %d points in time' % (nf, nt))
        print('Spectrogram had shape %d, %d' % spectrogram.shape)
        return 0, 0, 0
    # Z-score the flattened spectrogram
    if Norm:
        spectrogram -= spectrogram.mean()
        spectrogram /= spectrogram.std()
    if window == None:
        window = dt[-1]/10.0
    # Find the number of spectrogram points in the gaussian window 
    if dt[-1] < window:
        print('Warning in mps: window size is smaller or equal to spectrogram temporal extent.')
        print('mps will be calculate with a single window')
        nWindow = nt - 1
    else:
        nWindow = np.linspace(0, dt.size, dt.size)[dt>= window][0]
    if nWindow % 2 == 0:
        nWindow += 1  # Make it odd size so that we have a symmetric window
    if nWindow < 64:
        print('Error in mps: window size %d pts (%f s) is two small for reasonable estimates' % (nWindow, window))
        return 0, 0, 0        
    # Generate the Gaussian window
    gt, w = gaussian_window(nWindow, 6)
    tShift = int(gt[-1]/3)
    nchunks = 0    
    for tmid in range(tShift, nt, tShift):        
        # No zero padding at this point this could be better
        tstart = tmid-(nWindow-1)/2-1
        if tstart < 0:
            continue                       
        tend = tmid+(nWindow-1)/2
        if tend > nt:
            break
        nchunks += 1        
        # Multiply the spectrogram by the window
        wSpect = spectrogram[:,int(np.round(tstart)):int(np.round(tend)+1)]
        for fInd in range(nf):
            wSpect[fInd,:] = wSpect[fInd,:]*w            
        # Get the 2d FFT
        wf, wt, mps_pow,mps_phase = mtfft(wSpect, df, dt[int(tstart):int(tend)])
        if nchunks == 1:
            mps_powAvg = mps_pow
        else:
            mps_powAvg += mps_pow            
    mps_powAvg /= nchunks    
    return wf, wt, mps_powAvg

def get_mps(t, freq, spec):
    "Computes the MPS of a spectrogram (idealy a log-spectrogram) or other REAL time-freq representation"
    mps = np.fft.fftshift(np.fft.fft2(spec))
    amps = np.real(mps * np.conj(mps))
    nf = mps.shape[0]
    nt = mps.shape[1]
    wfreq = np.fft.fftshift(np.fft.fftfreq(nf, d=freq[1] - freq[0]))
    wt = np.fft.fftshift(np.fft.fftfreq(nt, d=t[1] - t[0]))
    return wt, wfreq, mps, amps

def inverse_mps(mps):
    "Inverts a MPS back to a spectrogram"
    spec = np.fft.ifft2(np.fft.ifftshift(mps))
    return spec

def mps_simple(strf, fstep, tstep, half=False):
    """Calculate the Modulation Power Spectrum of a STRF.
    Parameters
    ----------
    strf : array, shape (nfreqs, nlags)
        The STRF we'll use for MPS calculation.
    fstep : float
        The step size of the frequency axis for the STRF
    tstep : float
        The step size of the time axis for the STRF.
    half : bool
        Return the top half of the MPS (aka, the Positive
        frequency modulations)
    Returns
    -------
    mps_freqs : array
        The values corresponding to spectral modulations, in cycles / octave
        or cycles / Hz depending on the units of fstep
    mps_times : array
        The values corresponding to temporal modulations, in Hz
    amps : array
        The MPS of the input strf
    """
    # Convert to frequency space and take amplitude
    nfreqs, nlags = strf.shape
    fstrf = np.fliplr(strf)
    mps = np.fft.fftshift(np.fft.fft2(fstrf))
    amps = np.real(mps * np.conj(mps))

    # Obtain labels for frequency axis
    mps_freqs = np.zeros([nfreqs])
    fcircle = 1.0 / fstep
    for i in range(nfreqs):
        mps_freqs[i] = (i/float(nfreqs))*fcircle
        if mps_freqs[i] > fcircle/2.0:
            mps_freqs[i] -= fcircle

    mps_freqs = np.fft.fftshift(mps_freqs)
    if mps_freqs[0] > 0.0:
        mps_freqs[0] = -mps_freqs[0]

    # Obtain labels for time axis
    fcircle = tstep
    mps_times = np.zeros([nlags])
    for i in range(nlags):
        mps_times[i] = (i/float(nlags))*fcircle
        if mps_times[i] > fcircle/2.0:
            mps_times[i] -= fcircle

    mps_times = np.fft.fftshift(mps_times)
    if mps_times[0] > 0.0:
        mps_times[0] = -mps_times[0]

    if half:
        halfi = np.where(mps_freqs == 0.0)[0][0]
        amps = amps[halfi:, :]
        mps_freqs = mps_freqs[halfi:]

    return mps_freqs, mps_times, amps


def imps_simple(mps, out_shape=None):
    """Invert an MPS back to a spectrogram.
    Parameters
    ----------
    mps : array
        The MPS to be inverted with a 2D IFFT
    out_shape : array
        The dimensions of the output spectrogram
    Returns
    -------
    ispec : array, shape == out_shape
        The inverted spectrogram
    """
    mps = np.fft.ifftshift(mps)
    spec_i = np.real(np.fft.irfft2(mps, s=out_shape))
    return spec_i


def filter_mps(mps, mpsfreqs, mpstime, flim, tlim):
    """Filter out frequencies / times (in cycles) of the MPS.
    Parameters
    ----------
    mps : array, shape (n_freqs, n_times)
        The modulation power spectrum of a spectrogram / STRF.
    mpsfreqs : array, shape (n_freqs,)
        The values on the frequency (y) axis of the MPS
    mpstimes : array, shape (n_times,)
        The values on the time (x) axis of the MPS
    flim : array of floats | None, shape (2,)
        The minimum/maximum value on the frequency axis to keep
    tlim : array of floats | None, shape (2,)
        The minimum/maximum value on the time axis to keep
    Returns
    -------
    mps : array, shape (n_freqs, n_times)
        The MPS with various regions zeroed out
    msk_mps : array, shape (n_freqs, n_times)
        A boolean mask showing the frequencies kept
    """
    mps = mps.copy()
    msk_freq, msk_time = [np.zeros_like(mps, dtype=bool)
                      for _ in range(2)]
    # Mask for time axis
    use_times = mne.utils._time_mask(mpstime, *tlim)
    msk_time[:, use_times] = True
    # Mask for freq axis
    use_freqs = mne.utils._time_mask(mpsfreqs, *flim)
    msk_freq[use_freqs, :] = True
    # Combine masks for joint
    msk_mps = msk_time * msk_freq
    msk_remove = ~msk_mps

    # Create random phases for the filtered parts
    nphase = np.sum(msk_remove)
    phase_rand = 2 * np.pi * (np.random.rand(nphase) - .5)
    phase_rand = np.array([np.complex(0, i) for i in phase_rand])
    phase_rand = np.exp(phase_rand)

    # Now convert the masked values to 0 amplitude and rand phase
    mps[msk_remove] = 0. * phase_rand
    return mps, msk_mps

if __name__ == '__main__':
    from nsgt.cq import NSGT
    from transforms import ErbScale
    from matplotlib import pyplot as plt
    x, sr = librosa.load('blues.00000.au')
    modSpec = modspec(x, sr)
    plt.figure(figsize=(8,8))
    plt.imshow(np.log(modSpec))
    plt.axis('tight')
    print(modSpec.shape)
