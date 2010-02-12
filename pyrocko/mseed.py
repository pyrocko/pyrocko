import pyrocko.mseed_ext
from pyrocko.mseed_ext import HPTMODULUS, MSEEDERROR
import trace
import os
from util import reuse

def load(filename, getdata=True):

    mtime = os.stat(filename)[8]
    traces = []
    for tr in pyrocko.mseed_ext.get_traces( filename, getdata ):
        network, station, location, channel = tr[1:5]
        tmin = float(tr[5])/float(HPTMODULUS)
        tmax = float(tr[6])/float(HPTMODULUS)
        deltat = reuse(float(1.0)/float(tr[7]))
        ydata = tr[8]
        
        traces.append(trace.Trace(network, station, location, channel, tmin, tmax, deltat, ydata, mtime=mtime))
    
    return traces
    
def as_tuple(tr):
    itmin = int(round(tr.tmin*HPTMODULUS))
    itmax = int(round(tr.tmax*HPTMODULUS))
    srate = 1.0/tr.deltat
    return (tr.network, tr.station, tr.location, tr.channel, 
            itmin, itmax, srate, tr.get_ydata())

def save(traces, filename_template):
    fn_tr = {}
    for tr in traces:
        fn = tr.fill_template(filename_template)
        if fn not in fn_tr:
            fn_tr[fn] = []
        
        fn_tr[fn].append(tr)
        
    for fn, traces_thisfile in fn_tr.items():
        trtups = []
        traces_thisfile.sort(lambda a,b: cmp(a.full_id, b.full_id))
        for tr in traces_thisfile:
            trtups.append(as_tuple(tr))
            
        pyrocko.mseed_ext.store_traces(trtups, fn)
        
    return fn_tr.keys()
    
    
if __name__ == '__main__':
    import unittest
    import numpy as num
    import time
    import tempfile
    import random
    from random import choice as rc
    from os.path import join as pjoin

    abc = 'abcdefghijklmnopqrstuvwxyz' 
        
    def rn(n):
        return ''.join( [ random.choice(abc) for i in xrange(n) ] )
    
    class MSeedTestCase( unittest.TestCase ):
    
        def testWriteRead(self):
            now = time.time()
            n = 10
            deltat = 0.1
            
            networks = [ rn(2) for i in range(5) ]
            
            traces1 = [ trace.Trace(rc(networks), rn(4), rn(2), rn(3), tmin=now+i*deltat*n*2, deltat=deltat, ydata=num.arange(n), mtime=now)
                for i in range(100) ]
                
            tempdir = tempfile.mkdtemp()
            fns = save(traces1, pjoin(tempdir, '%(network)s'))
            traces2 = []
            for fn in fns:
                traces2.extend(load(fn))
                
            for tr in traces1:
                assert tr in traces2
                
            for fn in fns:
                os.remove(fn)
            
        def testReadNonexistant(self):
            try:
                trs = load('/tmp/thisfileshouldnotexist')
            except OSError, e:
                pass
            assert isinstance(e, OSError)
            
        def testReadEmpty(self):
            tempfn = tempfile.mkstemp()[1]
            try:
                trs = load(tempfn)
            except MSEEDERROR, e:
                pass
                
            assert str(e).find('No SEED data detected') != -1
            os.remove(tempfn)
        
    
    unittest.main()

        
