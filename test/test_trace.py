from pyrocko import trace, io, util
import unittest, math
import numpy as num

d2r = num.pi/180.

class TraceTestCase(unittest.TestCase):
    
    def testIntegrationDifferentiation(self):
        
        tlen = 100.
        dt = 0.1
        n = int(tlen/dt)
        f = 0.5
        tfade = tlen/10.
        
        xdata = num.arange(n)*dt
        ydata = num.sin(xdata*2.*num.pi*f)
        a = trace.Trace(channel='A', deltat=dt, ydata=ydata)
       
        b = a.transfer(tfade, (0.0, 0.1, 2.,3.), 
            transfer_function=trace.IntegrationResponse())
        b.set_codes(channel='B')
        
        c = a.transfer(tfade, (0.0, 0.1, 2.,3.), 
            transfer_function=trace.DifferentiationResponse())       
        c.set_codes(channel='C')
                
        eps = 0.001
        xdata = b.get_xdata()
        ydata = b.get_ydata()
        ydata_shouldbe = -num.cos(xdata*2*num.pi*f) /(2.*num.pi*f)
        assert num.amax(num.abs(ydata-ydata_shouldbe)) < eps, \
            'integration failed'
        
        xdata = c.get_xdata()
        ydata = c.get_ydata()
        ydata_shouldbe = num.cos(xdata*2*num.pi*f) *(2.*num.pi*f)
        assert num.amax(num.abs(ydata-ydata_shouldbe)) < eps, \
            'differentiation failed'
        
    def testDegapping(self):
        dt = 1.0
        atmin = 100.
        
        for btmin in range(90,120):
            a = trace.Trace(deltat=dt, ydata=num.zeros(10), tmin=atmin)
            b = trace.Trace(deltat=dt, ydata=num.ones(5), tmin=btmin)
            traces = [a,b]
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id) )
            xs = trace.degapper(traces)
            
            if btmin == 90:
                assert len(xs) == 2
            elif btmin > 90 and btmin < 115:
                assert len(xs) == 1
            else:
                assert len(xs) == 2
        
        a = trace.Trace(deltat=dt, ydata=num.zeros(10), tmin=100)
        b = trace.Trace(deltat=dt, ydata=num.ones(10), tmin=100)
        traces = [a,b]
        traces.sort( lambda a,b: cmp(a.full_id, b.full_id) )
        xs = trace.degapper(traces)
        assert len(xs) == 1
        for x in xs:
            assert x.tmin == 100
            assert x.get_ydata().size == 10
        
    def testRotation(self):
        s2 = math.sqrt(2.)
        ndata = num.array([s2,s2], dtype=num.float)
        edata = num.array([s2,0.], dtype=num.float)
        dt = 1.0
        n = trace.Trace(deltat=dt, ydata=ndata, tmin=100, channel='N')
        e = trace.Trace(deltat=dt, ydata=edata, tmin=100, channel='E')
        rotated = trace.rotate([n,e], 45., ['N','E'], ['R','T'])
        for tr in rotated:
            if tr.channel == 'R':
                r = tr
            if tr.channel == 'T':
                t = tr
        
        assert( num.all(r.get_ydata() - num.array([ 2., 1. ]) < 1.0e-6 ) )
        assert( num.all(t.get_ydata() - num.array([ 0., -1 ]) < 1.0e-6 ) )
            
    def testProjection(self):
        s2 = math.sqrt(2.)
        ndata = num.array([s2,s2], dtype=num.float)
        edata = num.array([s2,0.], dtype=num.float)
        ddata = num.array([1.,-1.], dtype=num.float)
        dt = 1.0
        n = trace.Trace(deltat=dt, ydata=ndata, tmin=100, channel='N')
        e = trace.Trace(deltat=dt, ydata=edata, tmin=100, channel='E')
        d = trace.Trace(deltat=dt, ydata=ddata, tmin=100, channel='D')
        azi = 45.
        cazi = math.cos(azi*d2r)
        sazi = math.sin(azi*d2r)
        rot45 = num.matrix([[cazi, sazi, 0],[-sazi,cazi, 0], [0,0,-1]], dtype=num.float)
        rotated = trace.project([n,e,d], rot45, ['N','E','D'], ['R','T','U'])
        for tr in rotated:
            if tr.channel == 'R':
                r = tr
            if tr.channel == 'T':
                t = tr
            if tr.channel == 'U':
                u = tr
        
        assert( num.all(r.get_ydata() - num.array([ 2., 1. ]) < 1.0e-6 ) )
        assert( num.all(t.get_ydata() - num.array([ 0., -1 ]) < 1.0e-6 ) )
        assert( num.all(u.get_ydata() - num.array([ -1., 1. ]) < 1.0e-6 ) )
        
        # should work though no horizontals given
        projected = trace.project([d], rot45, ['N','E','D'], ['R','T','U'])
        if tr.channel == 'U': u = tr 
        assert( num.all(u.get_ydata() - num.array([ -1., 1. ]) < 1.0e-6 ) )

        
    def testExtend(self):
        tmin = 1234567890.
        t = trace.Trace(tmin=tmin, ydata=num.ones(10,dtype=num.float))
        tmax = t.tmax
        t.extend(tmin-10.2, tmax+10.7)
        assert int(round(tmin-t.tmin)) == 10
        assert int(round(t.tmax-tmax)) == 10
        assert num.all(t.ydata[:10] == num.zeros(10, dtype=num.float))
        assert num.all(t.ydata[-10:] == num.zeros(10, dtype=num.float))
        assert num.all(t.ydata[10:-10] == num.ones(10, dtype=num.float))
        
        t = trace.Trace(tmin=tmin, ydata=num.arange(10,dtype=num.float)+1.)
        t.extend(tmin-10.2, tmax+10.7, fillmethod='repeat')
        assert num.all(t.ydata[:10] == num.ones(10, dtype=num.float))
        assert num.all(t.ydata[-10:] == num.zeros(10, dtype=num.float)+10.)
        assert num.all(t.ydata[10:-10] == num.arange(10, dtype=num.float)+1.)
    
    def testAppend(self):
        a = trace.Trace(ydata=num.zeros(0, dtype=num.float), tmin=1234567890)
        for i in xrange(10000):
            a.append(num.arange(1000, dtype=num.float))
        assert a.ydata.size == 10000*1000
        
        
if __name__ == "__main__":
    util.setup_logging('test_trace', 'warning')
    unittest.main()

