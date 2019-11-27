try:
    import pyfftw
except ImportError:
    pyfftw = None
else:
    pyfftw.interfaces.cache.enable()

class FFT(object):
    def __init__(self, n):
        if pyfftw is not None:
            self.cfwd = lambda x: pyfftw.interfaces.numpy_fft.fft(x, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=1)
            self.cbwd = lambda x: pyfftw.interfaces.numpy_fft.ifft(x, n=n, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=1)        
            self.rfwd = lambda x: pyfftw.interfaces.numpy_fft.rfft(x, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=1)
            self.rbwd = lambda x: pyfftw.interfaces.numpy_fft.irfft(x, n=n, overwrite_input=True, planner_effort='FFTW_ESTIMATE', threads=1)        
        else:
            import numpy as np
            self.cfwd = np.fft.fft
            self.cbwd = lambda x: np.fft.ifft(x, n=n)
            self.rfwd = np.fft.rfft
            self.rbwd = lambda x: np.fft.irfft(x, n=n)
