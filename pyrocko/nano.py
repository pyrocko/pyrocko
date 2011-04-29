from pyrocko import util
import time

def max_convertable():
    
    def test(i):
        return int(float(i)) == i
    
    i = 1
    while True:
        if not test(i-1):
            return i/2
        i*=2
    
INTFLOATMAX = max_convertable()

def convertable(i):
    return -INTFLOATMAX <= i <= INTFLOATMAX

G = 1000000000L

def asnano(x):
    if not isinstance(x, Nano):
        return Nano(x)
    else:
        return x
    
class RangeError(Exception):
    pass

class Nano(object):
    
    __slots__ = ('v',)

    def __init__(self, s=0, ns=0):
        self.v = int(s*G) + ns

    def __neg__(self):
        return Nano(ns=-self.v)

    def __pos__(self):
        return self

    def __add__(x,y):        
        return Nano(ns=x.v+asnano(y).v)

    __radd__ = __add__

    def __sub__(x,y):
        return x.__add__(-y)

    def __rsub__(x,y):
        return (-x).__add__(y)  

    def __mul__(x,y):
        return Nano(ns=x.v*asnano(y).v)

    __rmul__ = __mul__

    def __div__(x,y):
        return Nano(ns=x.v/asnano(y).v)

    def __rdiv__(x,y):
        return Nano(ns=asnano(y).v/x.v)

    def __mod__(x,y):
        return Nano(ns=x.v%asnano(y.v))

    def __rmod__(x,y):
        return Nano(ns=asnano(y).v%x.v)

    def __le__(x,y): 
        return x.v <= asnano(y).v

    def __ge__(x,y):
        return x.v >= asnano(y).v

    def __lt__(x,y):
        return x.v < asnano(y).v

    def __gt__(x,y):
        return x.v > asnano(y).v

    def __eq__(x,y):
        return x.v == asnano(y).v

    def __ne__(x,y):
        return x.v != asnano(y).v

    def __cmp__(x,y):
        return cmp(x.v,asnano(y).v)

    def __hash__(self):
        return hash(self.v)

    def __abs__(self):
        return Nano(ns=abs(self.v))

    def __int__(self):
        return self.v/G

    def __float__(self):
        if not convertable(self.v):
            # only about 100 days worth of seconds can be converted to a float
            raise RangeError('Nano type internal value %i is too large to be safely converted to a float.' % self.v)
        
        return float(self.v)/G

    def __str__(self):
        sign = '-+'[self.v >= 0]
        s = '%010i' % abs(self.v) 
        return '%s%s.%s' % (sign, s[:-9], s[-9:])



