from pyrocko import trace, pile, io, config, util

import unittest
import numpy as num
import tempfile, random, os
from random import choice as rc
from os.path import join as pjoin
    
def makeManyFiles( nfiles=200, nsamples=100000):
    

    abc = 'abcdefghijklmnopqrstuvwxyz' 
    
    def rn(n):
        return ''.join( [ random.choice(abc) for i in xrange(n) ] )
    
    stations = [ rn(4) for i in xrange(10) ]
    components = [ rn(3) for i in xrange(3) ]
    networks = [ 'xx' ]
    
    datadir = tempfile.mkdtemp()
    traces = []
    for i in xrange(nfiles):
        tmin = 1234567890+i*60*60*24*10 # random.randint(1,int(time.time()))
        deltat = 1.0
        data = num.ones(nsamples)
        traces.append(trace.Trace(rc(networks), rc(stations),'',rc(components), tmin, None, deltat, data))
    
    fnt = pjoin( datadir, '%(network)s-%(station)s-%(location)s-%(channel)s-%(tmin)s.mseed')
    io.save(traces, fnt, format='mseed')
    
    return datadir

class PileTestCase( unittest.TestCase ):
            
    def testPileTraversal(self):
        import tempfile, shutil
        config.show_progress = False
        nfiles = 200
        nsamples = 100000
        datadir = makeManyFiles(nfiles=nfiles, nsamples=nsamples)
        filenames = util.select_files([datadir])
        cachefilename = pjoin(datadir,'_cache_')
        p = pile.Pile(filenames, cachefilename)
        s = 0
        for traces in p.chopper(tmin=None, tmax=None, tinc=1234.): #tpad=10.):
            for trace in traces:
                s += num.sum(trace.ydata)
                
        os.unlink(cachefilename)
        shutil.rmtree(datadir)
        assert s == nfiles*nsamples
    

if __name__ == "__main__":
    pyrocko.util.setup_logging('warning')
    unittest.main()

