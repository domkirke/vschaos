import math
import time
import re
import os
import subprocess
import numpy as np
import scipy as sp
import soundfile as sf
import spectrum as spec
import matplotlib.pyplot as plt

'''
General Purpose Audio DSP functions

Should work on both Python 2.7.x and 3.x
Look at the test() function for some examples...

This collection is based upon the following packages:
  - numpy
  - scipy
  - pysoundfile
  - spectrum
  - matplotlib (for visualization)

This collection also requires the following software:
  - ffmpeg (optional, for mp3 handling)
  
'''

__version__ = "0.03"
__author__  = "G.Presti"


'''
------------------------
'''

def rms( x ):
    '''Compute RMS of each column'''
    return np.sqrt(np.mean(x**2, axis=0))

def peak( x ):
    '''Compute peak of each column'''
    return np.amax(np.fabs(x),axis=0)

def db2amp(x):
    '''Convert dB value to amp'''
    return 10**(x/20)

def amp2db( x, thres=-np.inf ):
    '''Convert amp value to dB'''
    t = db2amp(thres)
    return 20 * np.log10( np.maximum(x,t) )

eps = db2amp(-180)


def mp32wav( infile, outfile, frames=-1, start=0 ):
    '''Convert an mp3 to a wav file using ffmpeg'''
    '''additional params let you choose the number of frames and starting offset'''
    '''WARNING: will overwrite existing files without asking!'''

    if not os.path.isfile( infile ):
        raise Exception("Cannot find {0}".format( infile ))
    
    if (frames+start <= 0):
        cmd = 'ffmpeg -nostdin -y -i "' + infile + '" "' + outfile + '"'
    else:        
        Fs = audioinfo(infile).Fs
        t = float(frames) / Fs
        ss = float(start) / Fs
        cmd = 'ffmpeg -nostdin -y -ss ' + str(ss) + ' -t ' + str(t) + ' -i "' + infile + '" "' + outfile + '"'
        
    try:
        err = 0
        subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        err = e.returncode
    if err != 0:
        raise Exception("Error executing {0}\n\rffmpeg may be missing".format(cmd))    

def audioinfo( infile ):
    '''Get audio file stream info using ffprobe'''
    '''returns an object with attributes Fs, ch, bit, duration, length, codec'''
    '''WARNING: info.bit = 0 in case of lossy codec'''
    
    if not os.path.isfile( infile ):
        raise Exception("Cannot find {0}".format( infile ))
    
    try:
        err = 0
        cmd = 'ffprobe -show_streams "' + infile + '"'
        ffout = subprocess.check_output(cmd, shell=True).decode("utf-8")
    except subprocess.CalledProcessError as e:
        err = e.returncode
    if err != 0:
        raise Exception("Error executing {0}\n\rffprobe may be missing".format(cmd))

    class Info: pass
    info = Info()
    info.Fs = float(re.search('sample_rate=(.*)\\r',ffout).group(1))
    info.ch = int(re.search('channels=(.*)\\r',ffout).group(1))
    info.duration = float(re.search('duration=(.*)\\r',ffout).group(1))
    info.length = int(re.search('duration_ts=(.*)\\r',ffout).group(1))
    info.codec = re.search('codec_name=(.*)\\r',ffout).group(1)
    info.bit = int(re.search('bits_per_sample=(.*)\\r',ffout).group(1))   
    return info

def audioread( filename, frames=-1, start=0, fill_val=None ):
    '''Returns audio data in range -1:1 together with sample rate'''
    '''Additional params: start, frames, fill_value'''
    '''WARNING: mp3 read may end in len(data)!=frames'''
    if (filename[-3:].lower() == 'mp3'):
        fileDoExist = True
        while fileDoExist:
            temp_id = str(np.random.randint(100000,999999))
            tempfile = filename[:-4] + '_' + temp_id + '.wav'
            fileDoExist = os.path.isfile(tempfile)
        mp32wav(filename, tempfile, frames, start)
        data, Fs = sf.read(tempfile)
        os.remove(tempfile)
    else:
        data, Fs = sf.read(filename, frames, start, fill_value = fill_val)
    return data, Fs

def audiowrite( filename, data, Fs, subtype=None ):
    '''Write audio data into file'''
    '''Additional params: subtype'''
    '''PCM_24': 'Signed 24 bit PCM'''
    '''PCM_16': 'Signed 16 bit PCM'''
    '''PCM_S8': 'Signed 8 bit PCM'''
    '''WARNING: will overwrite existing files without asking!'''
    sf.write(filename,data,Fs,subtype)

def buffer( x, framesize, hopsize ):
    '''split an array into columns'''
    if len(x) % framesize != 0:
        xpad = np.append(x,np.zeros(framesize))
    else:
        xpad = x
    return np.asarray([xpad[i:i+framesize]for i in range(0, len(xpad)-framesize, hopsize)]).T

	
def unbuffer( buf, hopsize, w=1, ln=-1, fades = True ):
    '''Overlap and add columns'''
    '''w is the window function and ln is the final length'''
    framesize, n = buf.shape
    l = framesize + hopsize * (n-1)
    x = np.zeros(l)
    e = np.zeros(l)
    for n,i in enumerate(range(0, 1+l-framesize, hopsize)):
        x[i:i+framesize] += buf[:,n]
        e[i:i+framesize] += w
    e[e==0] = 0.1
    x = x[:ln]
    e = e[:ln]
    if fades:
        return fade(x/e, hopsize)
    else:
        return x/e

def stft( x, hopsize = None, Fs = None ):
    '''Real FFT along columns'''
    '''Also returns frequency and time axis'''
    wsize, n = x.shape
    X = np.fft.rfft(x, axis=0)
    if Fs is not None:
        F = np.fft.rfftfreq(wsize, 1.0/Fs)
        T = np.linspace(0,n-1,n)*hopsize/Fs
        return X, F, T
    else:
        return X

def istft( X ):
    '''Real iFFT along columns'''
    return np.fft.irfft(X, axis=0)

def finvert(X, hsiz, win = None):
    '''Invert magnitude-only stft'''
    fsiz, n = X.shape
    fsiz = 2 * (fsiz-1)
    if win is None: win = np.ones(fsiz)
    l = fsiz + hsiz * (n-1)
    p = np.random.randn(l+1)
    p = buffer(p,fsiz,hsiz)[:,0:n]
    p = p * win[:,None]
    return istft(stft(p)*X)
    

def armodel( x, order, method='yule' ):
    '''Model each column of the input according to order and method'''
    m = {
        'yule': spec.aryule,
        'burg': spec.arburg,
        'modcovar': spec.modcovar,
    }.get(method.lower())
    if x.ndim > 1:
        ar0 = np.ones(x.shape[1])
        model = np.asarray([m(x[:,c], order)[0] for c in range(x.shape[1])]).T
        return np.vstack([ar0,model])
    else:
        model = m(x, order)
        return np.append(1,model)

def ar2psd( ar, n ):
    '''Compute PSD from a collection of ar models'''
    if ar.ndim > 1:
        return np.asarray([spec.arma2psd(ar[1:,c], NFFT=2*(n-1))[:n] for c in range(ar.shape[1])]).T
    else:
        return spec.arma2psd(ar[1:], NFFT=2*(n-1))[:n]

def filters(b, a, data):
    '''Filter each column of x using parameters taken from corresponding column of b and a'''
    a = np.asarray(a)
    b = np.asarray(b)
    x = np.copy(np.asarray(data)) # Just to be sure, will be removed soon
    if a.ndim + b.ndim + x.ndim == 3:
        return sp.signal.lfilter(b,a,x)
    elif a.ndim + b.ndim == 2:
        return sp.signal.lfilter(b,a,x,axis=0)
    if x.ndim == 1:
        if a.ndim == 1:
            x = np.tile(x,[1,b.shape[1]])
        else:
            x = np.tile(x,[1,a.shape[1]])
    if b.ndim == 1:
        b = np.tile(b,[1,x.shape[1]])
    if a.ndim == 1:
        a = np.tile(a,[1,x.shape[1]])
    return np.asarray([sp.signal.lfilter(b[:,c],a[:,c],x[:,c]) for c in range(x.shape[1])]).T

def ar2poles(ar):
    '''Convert AR coefficients to complex poles'''
    rows, cols = ar.shape
    b = np.zeros(rows)
    b[0] = 1
    return np.asarray([spec.tf2zp(b,ar[:,c])[1] for c in range(cols)]).T

def poles2ar(p):
    '''Convert complex poles to AR coefficients'''
    rows, cols = p.shape
    z = 0
    k = 1
    return np.asarray([sp.signal.zpk2tf(z,p[:,c],k)[1] for c in range(cols)]).T

def z2f(p,Fs):
    '''Retrive poles/zeros frequency expressed in Hz'''
    fq = (Fs/2) * np.angle(p)/np.pi
    fq = np.sort(fq,axis=0)
    pks = fq.shape[0] // 2
    return fq[pks:,:]

def linf2logf(f, fref=440, nref=69, edo=12):
    '''Takes frequency values expressed in linear Hz to a log scale'''
    '''fref=440, nref=69, edo=12 for typical MIDI nn setup'''
    return edo * np.log2(f/fref) + nref

def logf2linf(nn, fref=440, nref=69, edo=12):
    '''Takes frequency values expressed in a log scale to linear Hz'''
    '''fref=440, nref=69, edo=12 for typical MIDI nn setup'''
    return np.exp2( (nn-nref)/edo ) * fref

def interpc(nx, xp, yp, method='linear'):
    '''Interpolation along columns'''
    '''method can be: linear, nearest, zero, slinear, quadratic, cubic'''
    worker = sp.interpolate.interp1d(xp,yp,axis=0,kind=method)
    return worker(nx)

def resample( x, iFs, oFs ):
    '''Resamples x from iFs sampling rate to oFs'''
    return sp.signal.resample_poly(x,oFs,iFs)  #Not very efficient

def fade(x, leng, typ = 'inout', shape = 2):
    '''apply fade-in, fade-out or both (typ =('in','out','inout')),
    of length len and exponent 'shape' '''
    typ = typ.lower()
    if typ == 'in' or typ == 'inout':
        x[0:leng] = x[0:leng] * (np.linspace(0,1,leng)**shape)
    if typ == 'out' or typ == 'inout':
        x[-leng:] = x[-leng:] * (np.linspace(1,0,leng)**shape)
    return x

'''
------------------------
'''

def test( filename, frames=-1, start=0 ):
    '''Test function'''

    '''Load sample file (convert to mono if necessary)'''
    x, Fs = audioread(filename, frames, start)
    if x.ndim > 1:
        x = np.mean(x, axis=1)




    '''Start timing...'''
    start = time.clock()

    '''Take to a lower sample rate'''
    x = resample(x, Fs, 16000)
    Fs = 16000

    '''Setup buffering variables and windowing function'''
    l = x.size
    fsiz = 2048
    hsiz = fsiz // 4
    win = np.hanning(fsiz)

    '''Split input into cunks (columns) and window them'''
    buf = buffer(x,fsiz,hsiz)
    buf = buf * win[:,None]

    '''Save input RMS'''
    env = rms(buf)

    '''Generate some noise (and split it into chunks too)'''
    noise = np.random.randn(l)
    noise = buffer(noise,fsiz,hsiz)
    noise = noise * win[:,None]

    '''Get the AR model of each input chunk (LPC)'''
    ar = armodel(buf, math.floor(2+Fs/1000), 'modcovar')

    '''Filter noise with the AR models and input with the inverse models'''
    '''(basically taking formants away from the input and putting them onto the noise)'''
    res = filters(ar, [1], buf)
    fmn = filters([1], ar, noise)

    '''Correct RMS to match that of the input'''
    res = res * env/rms(res)
    fmn = fmn * env/rms(fmn)

    '''Overlap and Add chunks to get the output signals'''
    fmn = fmn * win[:,None]
    fmn = unbuffer(fmn, hsiz, w=win**2, ln=l)
    res = res * win[:,None]
    res = unbuffer(res, hsiz, w=win**2, ln=l)

    '''Get the FT of the input chunks (with time and freq axes)'''
    X, F, T = stft(buf,hsiz,Fs)

    '''Some FFT resynthesis (2x time stretch)'''
    sin_res = finvert(np.abs(X),hsiz*2,win)
    sin_res = unbuffer(sin_res, hsiz*2, w=win, ln=l*2)

    '''Check how long it took'''
    sec = time.clock() - start
    print('Time: {} sec'.format(sec))




    '''Save the filtered input and noise'''
    audiowrite('Formant.wav',fmn/peak(fmn),Fs,subtype='PCM_24')
    audiowrite('Residual.wav',res/peak(res),Fs,subtype='PCM_24')
    audiowrite('Resint.wav',sin_res/peak(sin_res),Fs,subtype='PCM_24')




    '''Now let's plot some data...'''
    f, axarr = plt.subplots(2, sharex=True)

    '''Resamples the STFT in log scale (semitones)'''
    nA = logf2linf(np.linspace(30,linf2logf(4*Fs/10),1024))
    lX = interpc(nA,F,amp2db(np.abs(X),-40),method='slinear')

    '''Plot it'''
    axarr[0].imshow(lX,cmap="inferno",interpolation="nearest",origin="lower",aspect="auto",extent=[T[0],T[-1],linf2logf(nA[0]),linf2logf(nA[-1])])
    axarr[0].title.set_text ('Spectrum (log scale)')
    axarr[0].set_ylabel ('Semitones')    
    axarr[0].grid('on')
    
    '''Get the magnitude response of each AR model'''
    tfmag = np.sqrt(ar2psd(ar,1024))

    '''Plot it'''
    axarr[1].imshow(amp2db(tfmag/np.amax(tfmag),-74),cmap="inferno",interpolation="nearest",origin="lower",aspect="auto",extent=[T[0],T[-1],F[0],F[-1]])
    axarr[1].title.set_text ('Formants')
    axarr[1].set_ylabel ('Frequency (Hz)')
    axarr[1].set_xlabel ('Time (sec)')
    
    '''Computes the poles of the AR models'''
    p = ar2poles(ar)

    '''Find the frequency of each pole'''
    frqs = z2f(p,Fs)

    '''Plot the poles ontop of the AR models frequency response'''
    axarr[1].scatter( np.tile(T,(frqs.shape[0],1)) , frqs, s = 1, edgecolors='none' )
    axarr[1].grid('on')
    plt.show()




	
if __name__ == '__main__':
    '''Run the test case scenario'''
    '''First, select a file to analize'''
    testfile = './Samples/Perche.mp3'
    '''Get some info'''
    info = audioinfo(testfile)
    '''Select the rnge to analyse'''
    frames = 8.1 * info.Fs
    start = 10.4 * info.Fs
    '''Run test (downsample, LPC, Plots, wav file output)'''
    test(testfile,frames,start)

