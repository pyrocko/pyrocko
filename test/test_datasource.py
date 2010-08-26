#!/usr/bin/env python
import time
import random
import sys

deltat = 0.05
tbegin = time.time()
n = 0
while True:
    t = time.time() - tbegin
    nt = t/deltat
    while n < nt:
        d = random.randint(-127, 128)
        sys.stdout.write( "%i\n" % d )
        n += 1    
    
    sys.stdout.flush()
    tsleep = max(0.02, random.random()**10)
    time.sleep(tsleep)
    