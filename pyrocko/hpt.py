from pyrocko import util
import time

M = 1000000000L

def asnt(x):
    if not isinstance(x, NanoTime):
        return NanoTime(x)
    else:
        return x

class NanoTime(object):
    
    __slots__ = ('v',)

    def __init__(self, s=0, ns=0): 
        self.v = int(s*M) + ns

    def __neg__(self):
        return NanoTime(ns=-self.v)

    def __pos__(self):
        return self

    def __add__(x,y):        
        return NanoTime(ns=x.v+asnt(y).v)

    __radd__ = __add__

    def __sub__(x,y):
        return x.__add__(-y)

    def __rsub__(x,y):
        return (-x).__add__(y)  

    def __mul__(x,y):
        return NanoTime(ns=x.v*asnt(y).v)

    __rmul__ = __mul__

    def __div__(x,y):
        return NanoTime(ns=x.v/asnt(y).v)

    def __rdiv__(x,y):
        return Nanotime(ns=asnt(y).v/x.v)

    def __mod__(x,y):
        return NanoTime(ns=x.v%asnt(y.v))

    def __rmod__(x,y):
        return NanoTime(ns=asnt(y).v%x.v)

    def __le__(x,y): 
        return x.v <= y.v

    def __ge__(x,y):
        return x.v >= y.v

    def __lt__(x,y):
        return x.v < y.v

    def __gt__(x,y):
        return x.v > y.v

    def __eq__(x,y):
        return x.v == y.v

    def __ne__(x,y):
        return x.v != y.v

    def __cmp__(x,y):
        return cmp(x.v,y.v)

    def __hash__(x):
        return hash(x.v)

    def __abs__(x):
        return NanoTime(ns=abs(x.v))

    def __str__(self):
        return '%s.%09i' % (util.time_to_str(self.v/M, format="%Y-%m-%d %H:%M:%S"), self.v%M) 


print NanoTime(1,-1)
print NanoTime(1,1)
print NanoTime(-1,1)
print NanoTime(-1,1000000000)
print NanoTime(time.time())

print

print NanoTime(1) + 1.0
print 1.0 + NanoTime(1)

print NanoTime(3) - 1.0
print 3 - NanoTime(1)
print

print NanoTime(int(time.time())) + 1e-9
print +NanoTime(1)
a = NanoTime(1) 
a += 1
print a
