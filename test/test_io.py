from pyrocko import mseed, trace, util
import unittest
import numpy as num
import time
import tempfile
import random
from random import choice as rc
from os.path import join as pjoin
import os

abc = 'abcdefghijklmnopqrstuvwxyz' 
    
def rn(n):
    return ''.join( [ random.choice(abc) for i in xrange(n) ] )

class IOTestCase( unittest.TestCase ):

    def testWriteRead(self):
        now = time.time()
        n = 10
        deltat = 0.1
        
        networks = [ rn(2) for i in range(5) ]
        
        traces1 = [ trace.Trace(rc(networks), rn(4), rn(2), rn(3), tmin=now+i*deltat*n*2, deltat=deltat, ydata=num.arange(n, dtype=num.int32), mtime=now)
            for i in range(100) ]
            
        tempdir = tempfile.mkdtemp()
        fns = mseed.save(traces1, pjoin(tempdir, '%(network)s'))
        traces2 = []
        for fn in fns:
            traces2.extend(mseed.load(fn))
            
        for tr in traces1:
            assert tr in traces2
            
        for fn in fns:
            os.remove(fn)
        
    def testReadNonexistant(self):
        try:
            trs = mseed.load('/tmp/thisfileshouldnotexist')
        except OSError, e:
            pass
        assert isinstance(e, OSError)
        
    def testReadEmpty(self):
        tempfn = tempfile.mkstemp()[1]
        try:
            trs = mseed.load(tempfn)
        except mseed.MSeedError, e:
            pass
            
        assert str(e).find('No SEED data detected') != -1
        os.remove(tempfn)
    

if __name__ == "__main__":
    util.setup_logging('warning')
    unittest.main()

    
