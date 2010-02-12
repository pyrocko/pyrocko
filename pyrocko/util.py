import time
from scipy import signal


def decimate(x, q, n=None, ftype='iir', axis=-1):
    """downsample the signal x by an integer factor q, using an order n filter
    
    By default, an order 8 Chebyshev type I filter is used or a 30 point FIR 
    filter with hamming window if ftype is 'fir'.

    (port to python of the GNU Octave function decimate.)

    Inputs:
        x -- the signal to be downsampled (N-dimensional array)
        q -- the downsampling factor
        n -- order of the filter (1 less than the length of the filter for a
             'fir' filter)
        ftype -- type of the filter; can be 'iir' or 'fir'
        axis -- the axis along which the filter should be applied
    
    Outputs:
        y -- the downsampled signal

    """

    if type(q) != type(1):
        raise Error, "q should be an integer"

    if n is None:
        if ftype == 'fir':
            n = 30
        else:
            n = 8
    if ftype == 'fir':
        b = signal.firwin(n+1, 1./q, window='hamming')
        y = signal.lfilter(b, 1., x, axis=axis)
    else:
        (b, a) = signal.cheby1(n, 0.05, 0.8/q)
        y = signal.lfilter(b, a, x, axis=axis)

    return y.swapaxes(0,axis)[n/2::q].swapaxes(0,axis)

class UnavailableDecimation(Exception):
    pass
    
class Glob:
    decitab_nmax = 0
    decitab = {}

def mk_decitab(nmax=100):
    tab = Glob.decitab
    for i in range(1,10):
        for j in range(1,i+1):
            for k in range(1,j+1):
                for l in range(1,k+1):
                    for m in range(1,l+1):
                        p = i*j*k*l*m
                        if p > nmax: break
                        if p not in tab:
                            tab[p] = (i,j,k,l,m)
                    if i*j*k*l > nmax: break
                if i*j*k > nmax: break
            if i*j > nmax: break
        if i > nmax: break
    
def decitab(n):
    if n > Glob.decitab_nmax:
        mk_decitab(n*2)
    if n not in Glob.decitab: raise UnavailableDecimation('ratio = %g' % ratio)
    return Glob.decitab[n]

def gmctime(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    
def gmctime_v(t):
    return time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(t))

def gmctime_fn(t):
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(t))


reuse_store = dict()
def reuse(x):
    if not x in reuse_store:
        reuse_store[x] = x
    return reuse_store[x]
