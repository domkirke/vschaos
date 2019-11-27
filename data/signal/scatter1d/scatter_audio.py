#! /usr/bin/env python
# -*- coding: utf-8

import numpy as np
import logging
import os
import pickle
#from itertools import zip
from scatter1d import scatter, readaudio, reblock, synthesize, filters, filter_ds, scatplot
from scatter1d.util import nextpow2, isreal, issymm, pooling
from scatter1d.fft import FFT
from functools import reduce

def convertflt(flt, dtype=np.complex64, timedomain=False, threshold=1.e-6):
    """Convert filters from scatnet filter output pickle"""
    if isinstance(flt, dict):
        assert flt['type'] == 'fourier_truncated'
        assert flt['recenter']
        coefft = flt['coefft'][:,0]
        fltres = np.zeros(flt['N'], dtype=dtype)
        fltres[:len(coefft)] = coefft
        fltres = np.roll(fltres, flt['start']-1)
    elif isinstance(flt, np.ndarray):
        fltres = flt[...,0].astype(dtype)
    else:
        raise TypeError("Filter type not handled")
    
    assert isreal(fltres)
    fltres = fltres.real
    
    if timedomain:
        # output should be in time-domain, need to convert
        
        fltlen = len(fltres)
        if issymm(fltres):
            fltres = FFT(fltlen).rbwd(fltres[:fltlen//2+1])
            # time-domain fltres will be real
        else:
            fltres = FFT(fltlen).cbwd(fltres)
            # time-domain fltres will be complex
        # center filter kernel
        fltres = np.fft.fftshift(fltres)
        
        if threshold:
            fltabs = np.abs(fltres)
            absthresh = np.max(fltabs)*threshold
            fltexist = np.where(fltabs >= absthresh)[0]
            half1 = fltlen//2-np.min(fltexist)
            half2 = np.max(fltexist)+1-fltlen//2
            # half1 and half2 are most probably the same, since filter is symmetric
            half = nextpow2(max(half1,half2))
            # optimized time-domain kernel
            fltres = fltres[fltlen//2-half:fltlen//2+half]
        
    return fltres


def time2frames(time, sr=44100):
    """Convert different time formats to frame"""
    if ':' in time:
        # h:m:s
        secs = reduce(lambda a,b: float(a)*60+float(b), time.split(':'))
        time = int(secs/sr)        
    elif time.endswith('ms'):
        # milliseconds
        time = int(float(time[:-2])*0.001*sr)
    elif time.endswith('s'):
        # seconds
        time = int(float(time[:-1])*sr)
    else:
        # samples
        time = int(eval(time))
    return time

def scatter1D(audio, sr=44100, M=2, downsample=True, Q=[8,1], fmin=[20], fmax=[20000], synth_window='hanning', scatsize=4000, scathop=2000, filtersize=4000, filter_bw_thresh=10**(-6), poolingS=1024, log=False):
    '''usage: scatter_audio.py [-h] [--sr SR] [--outfile OUTFILE] [--M M]
                        [--downsample] [--no-downsample]
                        [--filters FILTERS | --Q Q] [--fmin FMIN]
                        [--fmax FMAX] [--pooling POOLING | --fps FPS]
                        [--poolfun POOLFUN] [--start START] [--stop STOP]
                        [--scatsize SCATSIZE] [--scathop SCATHOP]
                        [--synth-window SYNTH_WINDOW]
                        [--filtersize FILTERSIZE]
                        [--filter-bw-thresh FILTER_BW_THRESH]
                        [--feature-name FEATURE_NAME]
                        [--times-name TIMES_NAME] [--dtype DTYPE] [--log]
                        [--no-log] [--plot] [--no-plot] [--plotfile PLOTFILE]
                        [--figsize FIGSIZE]
                        infile
'''

        # set up logging
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO if log else logging.WARN)
    
    from time import time

    dtype = np.float32
    order = M
    fltbwthr = filter_bw_thresh

    samplerate = sr
    #infile = infile
    #fnpath, fnbase = os.path.split(infile)
    #fnbase_noext, fnext = os.path.splitext(fnbase)
    #fnparts = dict(fn=infile, path=fnpath, base=fnbase, base_noext=fnbase_noext, ext=fnext)
    
        ######################################################
    logging.info("Importing audio")
    #audio = readaudio(infile, samplerate=samplerate)
    audioframes = len(audio)

    #if args.start is None:
    #    start = 0
    #else:
    #    start = time2frames(args.start, sr=samplerate)

    #if args.stop is None:
    #    stop = audioframes
    #else:
    #    stop = time2frames(args.stop, sr=samplerate)

    # restrict audio
    # TODO: should be done in the reader/in streaming mode
    #audioframes = stop-start
    #audio = audio[start:start+audioframes]

    logging.info("Input audio length: %i samples" % len(audio))
    
    if scatsize is None:
        framesize = nextpow2(audioframes)
        hopsize = framesize
    else:
        #framesize = time2frames(scatsize, sr=samplerate)
        framesize = scatsize
        hopsize = scathop
        #if scathop.endswith("%"):
        #    hopsize = int(framesize*float(scathop[:-1])/100.)
        #elif args.scathop.endswith("x") or scathop.endswith("*"):
        #    hopsize = int(framesize/float(scathop[:-1]))
        #else:
        #    hopsize = time2frames(scathop, sr=samplerate)


    # pooling
    #if args.fps is not None:
    #    poolfactor = 2**int(np.log2(max(1, int(samplerate/args.fps))))
    #elif pooling:
    #poolframes = time2frames(pooling, sr=samplerate)
    poolframes = poolingS
    if poolframes:
        poolfactor = 2**int(np.log2(max(1, poolframes)))
    else:
        poolfactor = None
    #else:
    #    poolfactor = None
    
   # prepare filters
#if args.filters:
#    logging.info("Loading filters")
#    fltdata = np.load(args.filters)
#    k,flts = fltdata.iteritems().next()
#    k = cPickle.loads(k)

#    if order is None:
#        order = len(flts)
#    else:
#        order = min(order, len(flts))
        
#    phi = [convertflt(flt['phi']['filter'], threshold=fltbwthr) for _,flt,_ in flts[:order+1]]
#    psi = [[convertflt(f, threshold=fltbwthr) for f in flt['psi']['filter']] for _,flt,_ in flts[:order]]
#else:
    logging.info("Computing filters")
    phi = []
    psi = []
    #Q = map(int, Q.split(','))
    if not len(Q):
        parser.error("--Q must be given")
    while len(Q) < order+1:
        Q.append(Q[-1])

    if fmin is None:
        fmins = []
    else:
        fmins = [np.clip(float(f)/(samplerate/2.),0,samplerate/2.) for f in fmin]
    if not len(fmins):
        fmins = [None]
    while len(fmins) < order+1:
        fmins.append(fmins[-1])

    if fmax is None:
        fmaxs = []
    else:
        fmaxs = [np.clip(float(f)/(samplerate/2.),0,samplerate/2.) for f in fmax]
    if not len(fmaxs):
        fmaxs = [None]
    while len(fmaxs) < order+1:
        fmaxs.append(fmaxs[-1])
        
    if filtersize is None:
        filtersize = framesize
    else:
        filtersize = filtersize
        
    for q,fmin,fmax in zip(Q,fmins,fmaxs):
        # poolfactor*2 is for compatibility (--averaging) in scatnet_py
        phio, psio = filters(Q=q, n=filtersize, averaging=(poolfactor*2 if poolfactor else None), psi_min=fmin, psi_max=fmax, dtype=float)
        phi.append(phio)
        psi.append(psio)
    
    
    phi = phi[:order+1]
    psi = psi[:order]
            
    # compute downsampling factors
    dsf = int(downsample) # 0 or 1
    phi_ds = [filter_ds(phio, threshold=fltbwthr)**dsf for phio in phi]
    psi_ds = [[filter_ds(psiof, threshold=fltbwthr)**dsf for psiof in psio] for psio in psi]

    if poolfactor:
        # limit excessive downsampling factors
        phi_ds = [np.minimum(ds, poolfactor) for ds in phi_ds]
        psi_ds = [np.minimum(ds, poolfactor) for ds in psi_ds]
        
    # evaluate desired pooling functions
    poolfun = "mean,max"
    poolfuns = [eval("np.%s"%pf) for pf in poolfun.split(',')]
    poolfuns += [poolfuns[-1]]*max(0, order+1-len(poolfuns)) # add missing dimensions


    # anticipate number of resulting scattering frames
    nframes = 1+int(np.ceil(max(0, audioframes-framesize)/float(hopsize)))

    # reblock audio for (potentially overlapped) scattering frames
    stream = reblock((audio,), framesize, hopsize=hopsize, fullsize=True)
    
    ######################################################
    logging.info("Scattering...")
    
    scats = [list() for _ in range(order+1)]

    frametime = float(framesize)/samplerate

    time0 = time()
    for f,frame in enumerate(stream):
        s = scatter(frame, psi, phi, psi_ds, phi_ds)
        time1 = time()
        logging.info("Frame %i/%i, realtime/cputime=%.3g" % (f, nframes, (f+1)*frametime/(time1-time0)))
        for sr,si,pfun in zip(scats, s, poolfuns):
            if poolfactor:
                # perform averaging in addition to previous downsampling
                pf = poolfactor//(framesize//len(si))
                if pf > 1:
                    si = pooling(si, factor=pf, axis=0, poolfun=pfun)
            sr.append(si)

    logging.info("Scattering done (%i frames), realtime/cputime=%.3g" % (nframes, (f+1)*frametime/(time1-time0)))
    ######################################################
    
        # compute resulting downsampling factors
    scat_ds = [framesize//len(s[0]) for s in scats]
                    
    if scatsize:
        # synthesize overlapped frames to one matrix per order 
        wndfun = eval("np.%s"%synth_window)
        scats = [synthesize(s, hopsize=hopsize//ds, wndfun=wndfun, nframes=nframes) for s,ds in zip(scats, scat_ds)]
    else:
        scats = [s[0] for s in scats]
        
    # restrict number of resulting coefficients
    scats = [s[:int(np.ceil(float(audioframes)/ds))] for s,ds in zip(scats, scat_ds)]
    
    
        # report sizes
    for o,s in enumerate(scats):
        logging.info("Order %i, shape %s" % (o,s.shape))
    
    # ensure uniform size of results per order
    ncoeffs = reduce(lambda a,b: a*(a==b), map(len, scats))
    assert ncoeffs != 0

    # times for coefficients (border mode)
    times = np.arange(ncoeffs+1, dtype=dtype)*(float(scat_ds[0])/samplerate)
    
    return scats, times

    ######################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Apply scattering transform to audio in a frame-wise or monolithic manner')
    parser.add_argument("infile", type=str, help="Input audio (.wav or as suuported by ffmpeg/avconv) or data (.npy) file")
    parser.add_argument("--sr", type=float, default=44100, help="Sample rate (default=%(default)s)")
    parser.add_argument("--outfile", type=str, help="Output file template for scattering coefficients (optional, .h5 or .pkl)")
    # use of --filter above overrides the fltgroup options below
    parser.add_argument("--M", type=int, default=2, help="Scattering order (default=%(default)s)")
    parser.add_argument("--downsample", action='store_true', dest='downsample', help="Downsample if possible (default)")
    parser.add_argument("--no-downsample", action='store_false', dest='downsample', help="Do not attempt downsampling")
    parser.set_defaults(downsample=True)
    excgroup = parser.add_mutually_exclusive_group()
    excgroup.add_argument("--filters", type=str, help="Filter file as saved from scatnet_py (.pkl)")
    excgroup.add_argument("--Q", type=str, default="8,1", help="Q factors, comma-delimited for multiple orders (default=%(default)s)")
    parser.add_argument("--fmin", type=str, help="Minimum frequency for transform, comma-delimited for multiple orders (in Hz)")
    parser.add_argument("--fmax", type=str, help="Maximum frequency for transform, comma-delimited for multiple orders (in Hz)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pooling", type=str, default='1024', help="Result pooling (h:m:s,samples,*s,*ms, default=%(default)s)")
    group.add_argument("--fps", type=float, help="Resulting frames per seconds, tentative (default=%(default)s)")
    parser.add_argument("--poolfun", type=str, default='mean,max', help="Pooling function(s), comma-delimited for multiple orders (default=%(default)s)")
    parser.add_argument("--start", type=str, help="Start position in audio file (h:m:s,samples,*s,*ms, default=%(default)s)")
    parser.add_argument("--stop", type=str, help="Stop position in audio file (h:m:s,samples,*s,*ms, default=%(default)s)")
    parser.add_argument("--scatsize", type=str, help="Scattering frame size (h:m:s,samples,*s,*ms, default=%(default)s)")
    parser.add_argument("--scathop", type=str, default='50%', help="Scattering hop size (%%, *x, h:m:s,samples,*s,*ms, default='%(default)s')")
    parser.add_argument("--synth-window", type=str, default='hanning', help="Synthesis window (default='%(default)s')")
    parser.add_argument("--filtersize", type=str, help="Filter size (h:m:s,samples,*s,*ms, default equals --scatsize")
    parser.add_argument("--filter-bw-thresh", type=str, default='1.e-6', help="Filter bandwidth estimation threshold (default='%(default)s')")
    parser.add_argument("--feature-name", type=str, default='%(order)i', help="Feature name template in output file (default='%(default)s')")
    parser.add_argument("--times-name", type=str, default='times', help="Name of time data in data file (default='%(default)s')")
    parser.add_argument("--dtype", type=str, default='np.float32', help="dtype for results (default='%(default)s')")
    parser.add_argument("--log", action='store_true', dest='log', help="Log to console")
    parser.add_argument("--no-log", action='store_false', dest='log', help="Do not log to console (default)")
    parser.set_defaults(log=False)
    parser.add_argument("--plot", action='store_true', dest='plot', help="Plot results")
    parser.add_argument("--no-plot", action='store_false', dest='plot', help="Do not plot results (default)")
    parser.set_defaults(plot=False)
    parser.add_argument("--plotfile", type=str, help="Plot file template")
    parser.add_argument("--figsize", type=str, default="8,6", help="Figure size (default='%(default)s')")
    args = parser.parse_args()


    ######################################################
    






    if args.outfile:
        # write data file
        outdata = {args.feature_name%dict(order=ri):r.astype(dtype) for ri,r in enumerate(scats)}
        outdata[args.times_name] = times
        
        outfn = args.outfile%fnparts
        if outfn.endswith('.pck') or outfn.endswith('.pkl'):
            with file(outfn, 'wb') as f:
                logging.info("Saving results to %s"%outfn)
                cPickle.dump(outdata, f, protocol=cPickle.HIGHEST_PROTOCOL)
        elif outfn.endswith('.h5') or outfn.endswith('.hdf5'):
            import h5py
            with h5py.File(outfn, 'w') as f:
                for k,v in outdata.iteritems():
                    f[k] = v 

    if args.plot or args.plotfile:
        logging.info("Plotting")
        plotfile = args.plotfile%fnparts if args.plotfile else None
        figsize = map(float, args.figsize.split(','))
        scatplot(scats, times, audio=audio, samplerate=samplerate, title=fnbase, show=args.plot, savefig=plotfile, figsize=figsize)
