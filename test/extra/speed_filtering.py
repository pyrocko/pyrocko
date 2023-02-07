
import time
from pyrocko import trace
import numpy as num
from pyrocko import gmtpy


def timeit(f, duration=1.0):
    f()
    b = time.time()
    n = 0
    while (time.time() - b) < duration:
        f()
        n += 1
    return (time.time() - b)/n


def mktrace(n):
    tmin = 1234567890.
    t = trace.Trace(
        tmin=tmin, deltat=0.05, ydata=num.empty(n, dtype=float))
    return t


def bandpass(t):
    t.bandpass(4, 0.1, 5.)


def lowpass_highpass(t):
    t.lowpass(4,  5.)
    t.highpass(4, 0.1)


def bandpass_fft(t):
    t.bandpass_fft(0.1, 5.)


tab = []
for n in range(1, 22):
    a = timeit(lambda: bandpass(mktrace(2**n)))
    b = timeit(lambda: lowpass_highpass(mktrace(2**n)))
    c = timeit(lambda: bandpass_fft(mktrace(2**n)))
    print(2**n, a, b, c)
    tab.append((2**n, a, b, c))

a = num.array(tab).T
p = gmtpy.Simple()

for i in range(1, 4):
    p.plot((a[0], a[i]), '-W1p,%s' % gmtpy.color(i))

p.save('speed_filtering.pdf')
