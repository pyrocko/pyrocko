import time, logging, os, sys, re, calendar
from scipy import signal


from os.path import join as pjoin

import config

logger = logging.getLogger('pyrocko.util')

def setup_logging(programname, levelname,):
    levels = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}

    logging.basicConfig(
        level=levels[levelname],
        format = programname+':%(name)-20s - %(levelname)-8s - %(message)s' )

class Stopwatch:
    def __init__(self):
        self.start = time.time()
    
    def __call__(self):
        return time.time() - self.start
        
        
def progressbar_module():
    try:
        import progressbar
    except:
        logger.warn('progressbar module not available.')
        progressbar = None
    
    return progressbar


def progress_beg(label):
    if config.show_progress:
        sys.stderr.write(label)
        sys.stderr.flush()

def progress_end(label=''):
    if config.show_progress:
        sys.stderr.write(' done. %s\n' % label)
        sys.stderr.flush()
        
class GlobalVars:
    reuse_store = dict()
    decitab_nmax = 0
    decitab = {}
    decimate_fir_coeffs = {}
    decimate_iir_coeffs = {}
    
    

def decimate(x, q, n=None, ftype='iir'):
    """downsample the signal x by an integer factor q, using an order n filter
    
    By default, an order 8 Chebyshev type I filter is used or a 30 point FIR 
    filter with hamming window if ftype is 'fir'.

    (port to python of the GNU Octave function decimate.)

    Inputs:
        x -- the signal to be downsampled (1-dimensional array)
        q -- the downsampling factor
        n -- order of the filter (1 less than the length of the filter for a
             'fir' filter)
        ftype -- type of the filter; can be 'iir' or 'fir'
    
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
        coeffs = GlobalVars.decimate_fir_coeffs
        if (n, 1./q) not in coeffs:
            coeffs[n,1./q] = signal.firwin(n+1, 1./q, window='hamming')
        
        b = coeffs[n,1./q]
        y = signal.lfilter(b, 1., x)
    else:
        coeffs = GlobalVars.decimate_iir_coeffs
        if (n,0.05,0.8/q) not in coeffs:
            coeffs[n,0.05,0.8/q] = signal.cheby1(n, 0.05, 0.8/q)
           
        b, a = coeffs[n,0.05,0.8/q]
        y = signal.lfilter(b, a, x)

    return y[n/2::q].copy()

class UnavailableDecimation(Exception):
    pass
    
    
    
def gcd(a,b, epsilon=1e-7):
    while b > epsilon*a:
       a, b = b, a % b

    return a

def lcm(a,b):
    return a*b/gcd(a,b)

def mk_decitab(nmax=100):
    tab = GlobalVars.decitab
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
        
    GlobalVars.decitab_nmax = nmax
    
def decitab(n):
    if n > GlobalVars.decitab_nmax:
        mk_decitab(n*2)
    if n not in GlobalVars.decitab: raise UnavailableDecimation('ratio = %g' % ratio)
    return GlobalVars.decitab[n]

def ctimegm(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M:%S"))

def gmctime(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    
def gmctime_v(t):
    return time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime(t))

def gmctime_fn(t):
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(t))

def plural_s(n):
    if n == 1:
        return ''
    else:
        return 's' 

def ensuredirs(dst):
    d,x = os.path.split(dst)
    dirs = []
    while d and not os.path.exists(d):
        dirs.append(d)
        d,x = os.path.split(d)
        
    dirs.reverse()
    
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)

def ensuredir(dst):
    ensuredirs(dst)
    if not os.path.exists(dst):
        os.mkdir(dst)
    
def reuse(x):
    grs = GlobalVars.reuse_store
    if not x in grs:
        grs[x] = x
    return grs[x]
    
    
class Anon:
    def __init__(self,dict):
        for k in dict:
            self.__dict__[k] = dict[k]


def select_files( paths, selector=None,  regex=None ):

    progress_beg('selecting files...')
    if logger.isEnabledFor(logging.DEBUG): sys.stderr.write('\n')

    good = []
    if regex: rselector = re.compile(regex)

    def addfile(path):
        if regex:
            logger.debug("looking at filename: '%s'" % path) 
            m = rselector.search(path)
            if m:
                infos = Anon(m.groupdict())
                logger.debug( "   regex '%s' matches." % regex)
                for k,v in m.groupdict().iteritems():
                    logger.debug( "      attribute '%s' has value '%s'" % (k,v) )
                if selector is None or selector(infos):
                    good.append(os.path.abspath(path))
                
            else:
                logger.debug("   regex '%s' does not match." % regex)
        else:
            good.append(os.path.abspath(path))
        
        
    for path in paths:
        if os.path.isdir(path):
            for (dirpath, dirnames, filenames) in os.walk(path):
                for filename in filenames:
                    addfile(pjoin(dirpath,filename))
        else:
            addfile(path)
        
    progress_end('%i file%s selected.' % (len( good), plural_s(len(good))))
    
    return good

    

def base36encode(number, alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    '''
    Convert positive integer to a base36 string.
    '''
    if not isinstance(number, (int, long)):
        raise TypeError('number must be an integer')
    if number < 0:
        raise ValueError('number must be positive')
 
    # Special case for small numbers
    if number < 36:
        return alphabet[number]
 
    base36 = ''
    while number != 0:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36
 
    return base36
 
def base36decode(number):
    return int(number,36)

