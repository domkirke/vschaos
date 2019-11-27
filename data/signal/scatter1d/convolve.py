import numpy as np
from .util import nextpow2, isreal, issymm, resample
from .fft import FFT

class ConvKernel(object):
    def __init__(self, kernel, n, pad=None, convsize=None, dtype=None, timedomain=False):
        self.n = n
        self.dtype = dtype or kernel.dtype        
        ksize = len(kernel)
        
        if timedomain:
            minpad = max(n,ksize)
        else:
            minpad = ksize-1
            
        if pad is None:
            self.pad = minpad
        else:
            self.pad = pad
        self.coffs = 0
        self.csize = max(n, ksize)
        
        minconvsize = n+self.pad*2
        if convsize is None:
            self.convsize = nextpow2(minconvsize)
        else:
            self.convsize = convsize
        
        self.fft = FFT(self.convsize)
        
        self.sigpadded = np.zeros(self.convsize, dtype=self.dtype)
        
        kpadded = np.zeros(self.convsize, dtype=self.dtype)
        if timedomain:
            # assume time-domain kernel to be centered
            kpadded[:ksize] = kernel
            self.coffs = (min(ksize,n)-1)//2
            
            if np.isrealobj(kpadded):
                self.fkernel = self.fft.rfwd(kpadded)
            else:
                self.fkernel = self.fft.cfwd(kpadded)
        else:
            # padding in frequency domain amounts to kernel resampling to new length
            assert isreal(kernel)
            self.fkernel = resample(kernel.real, self.convsize)
            
            if issymm(self.fkernel):
                self.fkernel = self.fkernel[:len(self.fkernel)//2+1]


    def convolve(self, signal):
        if np.iscomplexobj(signal):
            # we are already transformed to spectral domain
            fsignal = signal
        else:
            # we are in the time domain, do transformation ad hoc
            assert len(signal)+self.pad <= self.convsize
            self.sigpadded[:self.pad] = 0
            self.sigpadded[self.pad:self.pad+len(signal)] = signal
            self.sigpadded[self.pad+len(signal):] = 0
            fsignal = self.fft.rfwd(self.sigpadded)
            
        # now, both fsignal and self.fkernel are in the spectral domain
        
        if (len(fsignal)-1)*2 == len(self.fkernel):
            # real * complex time signals
            klen = len(self.fkernel)
            ftmp = np.empty(klen, dtype=fsignal.dtype)
            ftmp[:klen//2+1] = fsignal # first half including 0 and Nyquist
            half2 = ftmp[klen//2+1:] 
            half2[:] = fsignal[klen//2-1:0:-1] # reverse second half excluding 0 and Nyquist
            np.conj(half2, out=half2)
            ftmp *= self.fkernel # complex multiplication
            conv = self.fft.cbwd(ftmp)
        elif len(fsignal) == len(self.fkernel):
            # real * real time signals
            np.multiply(fsignal, self.fkernel, out=fsignal)
            conv = self.fft.rbwd(fsignal)
        else:
            raise RuntimeError("Unexpected case")
            
        return conv[self.pad+self.coffs:self.pad+self.csize+self.coffs]


class ConvSignal(object):
    def __init__(self, signal, convsize, pad, dtype=None):
        self.dtype = dtype or signal.dtype
        self.convsize = convsize
        self.fft = FFT(self.convsize)        
        self.pad = pad
        sigpadded = np.zeros(self.convsize, dtype=self.dtype)
        self.siglen = len(signal)
        sigpadded[self.pad:self.pad+self.siglen] = signal
        self.fsignal = self.fft.rfwd(sigpadded)
        # self.fsignal has only half the complex coefficients + 1

    def convolve(self, kernel):
        assert kernel.convsize == self.convsize
        assert kernel.pad == self.pad
        res = kernel.convolve(self.fsignal)
        return res[:self.siglen]



# helper definitions
class Convolve(ConvKernel):
    def __init__(self, n, kernel, timedomain=True, dtype=float):
        super(Convolve, self).__init__(kernel, n, dtype=dtype, timedomain=timedomain)
    def __call__(self, signal):
        convsignal = ConvSignal(signal, convsize=self.convsize, pad=self.pad)
        return convsignal(self)


def convolve(signal, kernel):
    conv = Convolve(len(signal), kernel, dtype=signal.dtype, timedomain=True)
    return conv.convolve(signal)
