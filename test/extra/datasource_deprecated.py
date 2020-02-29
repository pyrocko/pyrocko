#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
import time
import random
import sys
import logging

from pyrocko import util

# simulate a 'bad' serial line data source, which produces samples at varying
# sampling rates, which may hang for some time, or produce samples at an
# inaccurate timing

logger = logging.getLogger('pyrocko.test.datasource')


def produce(deltat, duration):
    logging.debug('rate %g Hz, duration %g s' % (1./deltat, duration))
    tbegin = time.time()
    n = 0
    while True:
        t = time.time() - tbegin
        nt = int(t/deltat)
        while n < nt:
            d = random.randint(-127, 128)
            sys.stdout.write("%i\n" % d)
            n += 1

        sys.stdout.flush()

        tsleep = 0.01
        time.sleep(tsleep)

        if t > duration:
            break


util.setup_logging('producer', 'debug')

produce(0.0025, 20.)
logging.debug('sleep 2 s')
time.sleep(2.)
produce(0.0025, 20.)
produce(0.005, 20.)
produce(0.005005, 20.)
