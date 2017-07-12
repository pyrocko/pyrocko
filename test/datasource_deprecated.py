#!/usr/bin/env python
import time
import random
import sys

# simulate a 'bad' serial line data source, which produces samples at varying
# sampling rates, which may hang for some time, or produce samples at an
# inaccurate timing


def produce(deltat, duration):
    tbegin = time.time()
    n = 0
    while True:
        t = time.time() - tbegin
        nt = t/deltat
        while n < nt:
            d = random.randint(-127, 128)
            sys.stdout.write("%i\n" % d)
            n += 1

        sys.stdout.flush()

        tsleep = max(0.02, random.random()**10)
        time.sleep(tsleep)

        if t > duration:
            break


produce(0.025, 30.)
time.sleep(10)
produce(0.025, 30.)
produce(0.05, 30.)
produce(0.05005, 100.)
produce(0.1, 30.)
produce(0.05, 30.)
time.sleep(10.)
produce(0.05, 30.)
