# -*- coding: utf-8 -*-


import numpy as np
# - some functions adapted from gen_cort.m by Powen Ru (powen@isr.umd.edu), NSL, UMD

lastN, lastM = 0, 0

def filterSTRF(specgram, rates=[1, 2, 4, 8, 16, 32], scales=[0.5, 1, 2, 4, 8], fl=8, tc=8, fac=-2, shift=0, full_T=0, full_X=0, BP=0):
    """
    AUD2COR (forward) cortical rate-scale representation
    cr = aud2cor(y, para1, rv, sv, fname, DISP);
    cr    : cortical representation (4D, scale-rate(up-down)-time-freq.)
    y    : auditory spectrogram, N-by-M, where
    N = # of samples, M = # of channels
    para1 = [paras FULLT FULLX BP]
    paras    : [frmlen, tc, fac, shft] => frame length, time const, non-linear factor, fs octave shift
    FULLT (FULLX): fullness of temporal (spectral) margin. The value can
    be any real number within [0, 1]. If only one number was
    assigned, FULLT = FULLX will be set to the same value.
    BP    : pure bandpass indicator
    rv    : rate vector in Hz, e.g., 2.^(1:.5:5).
    sv    : scale vector in cyc/oct, e.g., 2.^(-2:.5:3).

    AUD2COR implements the 2-D wavelet transform
    possibly executed by the A1 cortex. The auditory
    spectrogram (Y) is the output generated by the 
    cochlear model (WAV2AUD) according to the parameter
    set PARA1. RV (SV) is the characteristic frequencies
    (ripples) of the temporal (spatial) filters. This
    function will store the output in a file with a
    conventional extension .COR. Roughly, one-second
    signal needs about 22 MB if 8-ms-frame is adopted.
    Choosing truncated fashion (FULL = 0) will reduce
    the size to 1/4, which will also reduce runing time
    to half.
    See also: WAV2AUD, COR_INFO, CORHEADR, COR_RST
    """

    K1 = len(rates)
    K2 = len(scales)
    (N, M) = specgram.shape
    global lastN, lastM
    lastN, lastM = N, M
    N1 = int(2**np.ceil(np.log2(N)))
    N2 = 2*N1
    M1 = int(2**np.ceil(np.log2(M)))
    M2 = 2*M1
    # 2D FT of specgram to perform rate/scale filter in secgram-freq domain
    Y = np.fft.fft2(specgram,s=(N2,M2))[:,0:M1]
    STF = 1000.0 / fl    # frame per second
    SRF = 24        # channel per octave (fixed)

    # freq. index
    dM   = int(float(M)/2*full_X)
    mdx1 = np.hstack((np.arange(dM)+M2-dM, np.arange(M)+dM))
    # temp. index
    dN   = int(float(N)/2*full_T)
    ndx  = np.arange(N)+2*dN
    ndx1 = ndx
    z  = np.zeros((N+2*dN, M+2*dM), dtype='complex128')
    cr = np.zeros((K2, K1*2, N+2*dN, M+2*dM), dtype='complex128')
    for rdx in range(K1):
        # rate filtering
        fc_rt = rates[rdx]
        HR = temporal_filter(fc_rt, N1, STF, [1+rdx+BP, K1+BP*2])
        for sgn in (1, -1):
            # rate filtering modification
            if sgn > 0:
                HR = np.hstack((HR, np.zeros(N1)))
            else:
                HR = np.hstack( (HR[0], np.conj(HR[N2:0:-1])) )
                if N2 > 2:
                	HR[N1] = np.abs(HR[N1+1])  
            # first inverse fft (w.r.t. time axis)
            z1 = HR[:,None] * Y
            z1 = np.fft.ifft(z1,axis=0)
            z1 = z1[ndx1,:]
            for sdx in range(K2):
                # scale filtering
                fc_sc = scales[sdx]
                HS = frequency_filter(fc_sc, M1, SRF, [1+sdx+BP, K2+BP*2])
            
                # second inverse fft (w.r.t frequency axis)
                z[ndx,:] = np.fft.ifft(z1*HS,axis=1,n=M2)[ndx[:,None],mdx1]
                cr[sdx, rdx+(sgn==1)*K1, :, :] = z
    return cr

def corfftc(z, Z_cum, N, M, HR, HS, HH):
    """
    CORFFTC cortical fft and cumulation 
    z: N-by-M complex matrix
    Z_cum: cumulated reverse response
    HR, HS: rate, scale transfer function
    HH: cumulated reverse filter transfer function
    CORFFTC is an internal routine to perform the 2-D reverse filtering
    and cumulate the response which be used to reconstruct the auditory
    spectrogram
    """
    # 2-D FFT
    Z = np.zeros((N[2], M[1]), dtype=complex)
    if M[3]:
        idLeft = np.concatenate((np.array(range(M[0]+M[3])), np.array(range(M[3]))+M[2]-M[3]))
        idRight = np.concatenate((np.array(range(M[0]+M[3]))+M[3], np.array(range(M[3]))))
        z[:, idLeft] = z[:, idRight];
    else:
        zTmp = np.zeros((z.shape[0], M[2]), dtype=complex)
        zTmp[:z.shape[0], :z.shape[1]] = z	
        z = zTmp
    for n in range(int(N[0]+N[3]*2)):
        R1 = np.fft.fft(z[n, :]);
        Z[n, :] = R1[:int(M[1])];
    for m in range(int(M[1])):
        Z[:, m] = np.fft.fft(Z[:, m]);
    # cumulation
    R1 = np.dot(HR[:, np.newaxis], HS[:, np.newaxis].T);
    HH = HH + R1 * np.conj(R1);
    Z_cum = Z_cum + R1 * Z;
    return Z_cum, HH

def cornorm(Z_cum, HH, N, M, norm=.9, foutt=0, foutx=0):
    """
    CORNORM cortical normalization and ifft
    Z_cum	: cumulated reverse response
    HH	: overall 2-D response
    N, M	: dimensions of the original auditory spectrum
    NORM	: 0=flat, 1=full, .x=partial normalization
    FOUTT(X): (optional) overlapped output, within [0, 1]
    CORNORM normalizes 2-D rate-scale response and then does
    the 2-D inverse fft to conclude reconstruction.
    """
    # Modify overall transfer function
    sumH = np.sum(HH);
    HH = norm * HH + (1 - norm) * np.max(HH);
    HH = HH / np.sum(HH) * sumH;
    # normalization
    ndx = np.array(range(int(N[2])))
    mdx = np.array(range(int(M[1])))
    #Z_cum[ndx][:, mdx] = Z_cum[ndx][:, mdx] / HH[ndx][:, mdx]
    Z_cum = Z_cum / HH
    N[3] = np.floor(N[0]/2*foutt);	
    ndx = np.array(range(int(N[0]+2*N[3])))
    ndx1 = np.concatenate((np.array(range(int(N[3])))+N[2]-N[3], np.array(range(int(N[0]+N[3]))))).astype(int)
    M[3] = np.floor(M[0]/2*foutx)
    mdx1 = np.concatenate((np.array(range(int(M[3])))+M[2]-M[3], np.array(range(int(M[0]+M[3]))))).astype(int)
    y = np.zeros((int(N[0]+2*N[3]), int(M[1])),dtype=complex)
    yh = np.zeros((int(N[0]+2*N[3]), int(M[0]+2*M[3])),dtype=complex)
    # 2-D IFFT
    for m in range(int(M[1])):
        R1 = np.fft.ifft(Z_cum[:, m]);
        y[:, m] = R1[ndx1];    
    for n in ndx:
        R1 = np.fft.ifft(y[n, :], int(M[2]))
        yh[n, :] = R1[mdx1];
    yh = yh * 2;
    return yh

def inverseSTRF(specgram, rates=[1, 2, 4, 8, 16, 32], scales=[0.5, 1, 2, 4, 8], fl=8, tc=8, fac=-2, shift=0, full_T=0, full_X=0, BP=0, norm=.9, foutt=0, foutx=0):
    """
    AUD2COR (forward) cortical rate-scale representation
    cr = aud2cor(y, para1, rv, sv, fname, DISP);
    cr    : cortical representation (4D, scale-rate(up-down)-time-freq.)
    y    : auditory spectrogram, N-by-M, where
    N = # of samples, M = # of channels
    para1 = [paras FULLT FULLX BP]
    paras    : [frmlen, tc, fac, shft] => frame length, time const, non-linear factor, fs octave shift
    FULLT (FULLX): fullness of temporal (spectral) margin. The value can
    be any real number within [0, 1]. If only one number was
    assigned, FULLT = FULLX will be set to the same value.
    BP    : pure bandpass indicator
    rv    : rate vector in Hz, e.g., 2.^(1:.5:5).
    sv    : scale vector in cyc/oct, e.g., 2.^(-2:.5:3).

    AUD2COR implements the 2-D wavelet transform
    possibly executed by the A1 cortex. The auditory
    spectrogram (Y) is the output generated by the 
    cochlear model (WAV2AUD) according to the parameter
    set PARA1. RV (SV) is the characteristic frequencies
    (ripples) of the temporal (spatial) filters. This
    function will store the output in a file with a
    conventional extension .COR. Roughly, one-second
    signal needs about 22 MB if 8-ms-frame is adopted.
    Choosing truncated fashion (FULL = 0) will reduce
    the size to 1/4, which will also reduce runing time
    to half.
    See also: WAV2AUD, COR_INFO, CORHEADR, COR_RST
    """
    # Retrieve parameters
    K1 = len(rates)
    K2 = len(scales)
    N, M = lastN, lastM
    N1 = int(2**np.ceil(np.log2(N)))
    N2 = 2*N1
    M1 = int(2**np.ceil(np.log2(M)))
    M2 = 2*M1
    STF	= 1000 / fl
    SRF	= 24;
    M4 = np.floor(float(M)/2*full_X);	# dM
    N4 = np.floor(float(N)/2*full_T);	# dN
    HH   = 0;
    Z_cum = 0;
    for rdx in range(K1):
        # rate filtering
        fc_rt = rates[rdx]
        HR = temporal_filter(fc_rt, N1, STF, [1+rdx+BP, K1+BP*2])
        for sgn in (1, -1):
            # rate filtering modification
            if sgn > 0:
                HR = np.conj(np.hstack((HR, np.zeros(N1))))
            else:
                HR = np.hstack( (HR[0], np.conj(HR[N2:0:-1])) )
                HR[N1] = np.abs(HR[N1+1])  
            for sdx in range(K2):
                # scale filtering
                fc_sc = scales[sdx]
                HS = frequency_filter(fc_sc, M1, SRF, [1+sdx+BP, K2+BP*2])
                z = np.squeeze(specgram[sdx, rdx+(sgn==1)*K1, :, :]).copy()
                # 2-D FFT and cumulation
                Z_cum, HH = corfftc(z, Z_cum, [N, N1, N2, N4], [M, M1, M2, M4], HR, HS, HH);
        #qlsdkjcn
    # normalization
    HH[:, 0] = HH[:, 0] * 2
    yh = cornorm(Z_cum, HH, [N, N1, N2, N4], [M, M1, M2, M4], norm, foutt, foutx);
    return yh

def temporal_filter(fc, L, srt, PASS = [2,3]):
    """Generate (bandpass) cortical filter transfer function
    fc: characteristic frequency
    L: filter length (use power of 2)
    srt: sample rate
    PASS: (vector) [idx K]
    idx = 1, lowpass; 1<idx<K, bandpass; idx = K, highpass.

    GEN_CORT generate (bandpass) cortical temporal filter for various
    length and sampling rate. The primary purpose is to generate 2, 4,
    8, 16, 32 Hz bandpass filter at sample rate ranges from, roughly
    speaking, 65 -- 1000 Hz. Note the filter is complex and non-causal.
    see also: AUD2COR, COR2AUD, MINPHASE
    """
    if issubclass(type(fc), str):
        fc = float(fc)
    t = np.arange(L).astype(np.float32)/srt
    k = t*fc
    h = np.sin(2*np.pi*k) * k**2 * np.exp(-3.5*k) * fc

    h = h-np.mean(h)
    H0 = np.fft.fft(h, n=2*L)
    A = np.angle(H0[0:L])
    H = np.abs(H0[0:L])
    maxi = np.argmax(H)
    H = H / (H[maxi] or 1)

    # passband
    if PASS[0] == 1:
        #low pass
        H[0:maxi] = 1
    elif PASS[0] == PASS[1]:
        #high pass
        H[maxi+1:L] = 1

    H = H * np.exp(1j*A)
    return H

def frequency_filter(fc, L, srf, KIND=2):
    """
    GEN_CORF generate (bandpass) cortical filter transfer function
    h = gen_corf(fc, L, srf);
    h = gen_corf(fc, L, srf, KIND);
    fc: characteristic frequency
    L: length of the filter, power of 2 is preferable.
    srf: sample rate.
    KIND: (scalar)
          1 = Gabor function; (optional)
          2 = Gaussian Function (Negative Second Derivative) (defualt)
          (vector) [idx K]
          idx = 1, lowpass; 1<idx<K, bandpass; idx = K, highpass.

    GEN_CORF generate (bandpass) cortical filter for various length and
    sampling rate. The primary purpose is to generate 2, 4, 8, 16, 32 Hz
    bandpass filter at sample rate 1000, 500, 250, 125 Hz. This can also
    be used to generate bandpass spatial filter .25, .5, 1, 2, 4 cyc/oct
    at sample ripple 20 or 24 ch/oct. Note the filter is complex and
    non-causal.
    see also: AUD2COR, COR2AUD
    """

    if hasattr(KIND, "__len__"):
        PASS = KIND
        KIND = 2
    else:
        PASS = [2,3]
        KIND = [KIND]

    # fourier transform of lateral inhibitory function 

    # tonotopic axis
    if issubclass(type(fc), str):
        fc = float(fc)
    R1 = np.arange(L).astype(np.float)/L*srf/2/np.abs(fc)

    if KIND == 1:
        # Gabor function
        C1      = 1./2/0.3/0.3
        H       = np.exp(-C1*(R1-1)**2) + np.exp(-C1*(R1+1)**2)
    else:
        # Gaussian Function
        R1    = R1 ** 2
        H    = R1 * np.exp(1-R1)

    # passband
    if PASS[0] == 1:
        #lowpass
        maxi = np.argmax(H)
        sumH = H.sum()
        H[0:maxi] = 1
        H = H / (H.sum()  or 1) * sumH
    elif PASS[0] == PASS[1]:
        # highpass
        maxi = np.argmax(H)
        sumH = H.sum()
        H[maxi+1:L] = 1
        H = H / (H.sum() or 1) * sumH

    return H

def strf(aud, fl=10, bp=1, rates=[1, 2, 4, 8, 16, 32], scales=[0.5, 1, 2, 4, 8], tc=8, fac=-2, shift=0, full_T=0, full_X=0):
    # Perform th
    cor = filterSTRF(aud,BP=bp,fl=fl, rates=rates, scales=scales, tc=tc, fac=fac, shift=shift, full_T=full_T, full_X=full_X)
    #collapse +/- rates
    #nr = cor.shape[1]
    #cor = np.abs(cor[:,0:int(nr/2),:,:]) + np.abs(cor[:,int(nr/2):nr,:,:])
    return cor #np.abs(cor)

def istrf(aud, fl=10, bp=1, rates=[1, 2, 4, 8, 16, 32], scales=[0.5, 1, 2, 4, 8], tc=8, fac=-2, shift=0, full_T=0, full_X=0):
    # Perform inverse STRF
    spec = inverseSTRF(aud,BP=bp,fl=fl, rates=rates, scales=scales, tc=tc, fac=fac, shift=shift, full_T=full_T, full_X=full_X)
    return spec