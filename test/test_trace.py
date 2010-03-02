from pyrocko import trace, io, util
import unittest, math
import numpy as num

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
        for atmin, btmin in [ (100., 90.), (100.,92.), (100.,97.), (100.,100.), (100.,102.), (100.,105.), (100.,107.), (100., 110.), (100.,114.), (100.,120.) ]:
            a = trace.Trace(deltat=dt, ydata=num.zeros(10), tmin=atmin)
            b = trace.Trace(deltat=dt, ydata=num.ones(5), tmin=btmin)
            traces = [a,b]
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id) )
            xs = trace.degapper(traces)
            for x in xs:
                print x.tmin, x.get_ydata()
            print '--'
        
        a = trace.Trace(deltat=dt, ydata=num.zeros(10), tmin=100)
        b = trace.Trace(deltat=dt, ydata=num.ones(10), tmin=100)
        traces = [a,b]
        traces.sort( lambda a,b: cmp(a.full_id, b.full_id) )
        xs = trace.degapper(traces)
        for x in xs:
            print x.tmin, x.get_ydata()
            
        print '--'
        
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
            
        
        
        
if __name__ == "__main__":
    util.setup_logging('warning')
    unittest.main()

