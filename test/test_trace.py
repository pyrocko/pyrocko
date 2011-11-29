from pyrocko import trace, io, util, model
import unittest, math, time
import numpy as num

sometime = 1234567890.
d2r = num.pi/180.

def numeq(a,b, eps):
    return num.all(num.abs(num.array(a) - num.array(b)) < eps)

def floats(l):
    return num.array(l, dtype=num.float)

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
        
        for meth, res in (('use_second', [1.,1.]), ('use_first', [0.,0.]), ('crossfade_cos', [0.25,0.75])):
            a = trace.Trace(deltat=dt, ydata=num.zeros(10), tmin=100)
            b = trace.Trace(deltat=dt, ydata=num.ones(10), tmin=108)
            traces = [a,b]
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id) )
            xs = trace.degapper(traces, deoverlap=meth)
            for x in xs:
                assert x.ydata.size == 18
                assert numeq(x.ydata[8:10], res, 1e-6)
                




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
        
        assert numeq(r.get_ydata(), [2.,1.], 1.0e-6)
        assert numeq(t.get_ydata(), [ 0., -1 ], 1.0e-6)
            
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
        rot45 = num.array([[cazi, sazi, 0],[-sazi,cazi, 0], [0,0,-1]], dtype=num.float)
        C = lambda x: model.Channel(x)
        rotated = trace.project([n,e,d], rot45, [C('N'),C('E'),C('D')], [C('R'),C('T'),C('U')])
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
        
        deps = trace.project_dependencies(rot45, [C('N'),C('E'),C('D')], [C('R'),C('T'),C('U')])
        assert( set(['N','E']) == set(deps['R']) and set(['N', 'E']) == set(deps['T']) and
                set(['D']) == set(deps['U']) )
        
        # should work though no horizontals given
        projected = trace.project([d], rot45, [C('N'),C('E'),C('D')], [C('R'),C('T'),C('U')])
        if tr.channel == 'U': u = tr 
        assert( num.all(u.get_ydata() - num.array([ -1., 1. ]) < 1.0e-6 ) )
        
        
    def testExtend(self):
        tmin = sometime
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
        a = trace.Trace(ydata=num.zeros(0, dtype=num.float), tmin=sometime)
        for i in xrange(10000):
            a.append(num.arange(1000, dtype=num.float))
        assert a.ydata.size == 10000*1000
        
    def testDownsampling(self):
        
        n = 1024
        dt1 = 1./125.
        dt2 = 1./10.
        dtinter = 1./util.lcm(1./dt1,1./dt2)
        upsratio = dt1/dtinter
        xdata = num.arange(n,dtype=num.float)
        ydata = num.exp(-((xdata-n/2)/10.)**2)
        t = trace.Trace(ydata=ydata, tmin=sometime, deltat=dt1, location='1')
        t2 = t.copy()
        t2.set_codes(location='2')
        t2.downsample_to(dt2, allow_upsample_max = 10)
        io.save([t,t2], 'test.mseed')
        
    def testFiltering(self):
        tmin = sometime
        b = time.time()
        for i in range(100):
            n = 10000
            t = trace.Trace(tmin=tmin, deltat=0.05, ydata=num.ones(n,dtype=num.float))
            t.lowpass(4,  5.)
            t.highpass(4, 0.1)
        d1 = time.time() - b
        b = time.time()
        for i in range(100):
            n = 10000
            t = trace.Trace(tmin=tmin, deltat=0.05, ydata=num.ones(n,dtype=num.float))
            t.bandpass_fft(0.1, 5.)
        d2 = time.time() - b
        
    def testCropping(self):
        n = 20
        tmin = sometime
        t = trace.Trace(tmin=tmin, deltat=0.05, ydata=num.ones(n,dtype=num.float))
        tmax = t.tmax
        t.ydata[:3] = 0.0
        t.ydata[-3:] = 0.0
        t.crop_zeros()
        assert abs(t.tmin - (tmin + 0.05*3)) < 0.05*0.01
        assert abs(t.tmax - (tmax - 0.05*3)) < 0.05*0.01
        t = trace.Trace(tmin=tmin, deltat=0.05, ydata=num.zeros(n,dtype=num.float))
        ok = False
        try:
            t.crop_zeros()
        except trace.NoData:
            ok = True
        
        assert ok
        
        t = trace.Trace(tmin=tmin, deltat=0.05, ydata=num.ones(n,dtype=num.float))
        t.crop_zeros()
        assert t.ydata.size == 20
        
        
    def testAdd(self):
        n = 20
        tmin = sometime
        deltat = 0.05
        for distortion in [ -0.2, 0.2 ]:
            for ioffs, result in [ (  1, [0.,1.,1.,1.,0.]),
                                ( -1, [1.,1.,0.,0.,0.]),
                                (  3, [0.,0.,0.,1.,1.]),
                                (  10, [0.,0.,0.,0.,0.]),
                                (  -10, [0.,0.,0.,0.,0.]) ]:
                                
                a = trace.Trace(tmin=tmin, deltat=deltat, ydata=num.zeros(5,dtype=num.float))
                b = trace.Trace(tmin=tmin+(ioffs+distortion)*deltat, deltat=deltat, ydata=num.ones(3,dtype=num.float))
                a.add(b, interpolate=False)
                assert numeq(a.ydata, result, 0.001)
        
    
    def testPeaks(self):
        n = 1000
        t = trace.Trace(tmin=0, deltat=0.1, ydata=num.zeros(n, dtype=num.float))
        t.ydata[1] = 1.
        t.ydata[500] = 1.
        t.ydata[999] = 1.
        tp, ap = t.peaks(0.5, 1.)
        assert numeq( tp, [0., 49.9, 99.8], 0.0001)
        assert numeq( ap, [1., 1., 1.], 0.0001)

        t.ydata[504] = 1.0 
        t.ydata[511] = 1.0
        
        tp, ap = t.peaks(0.5, 1.)

        assert numeq( tp, [0., 49.9, 51., 99.8], 0.0001)
        assert numeq( ap, [1., 1., 1., 1.], 0.0001)

    def testCorrelate(self):
        
        for la, lb, mode, res in [
            ([0, 1, .5, 0, 0 ],    [0, 0, 0, 1, 0 ],    'same', 0.3),
            ([0, 1, .5, 0, 0, 0 ], [0, 0, 0, 1, 0, 0 ], 'same', 0.3), 
            ([0, 1, .5, 0, 0 ],    [0, 0, 0, 1, 0, 0 ], 'same', 0.3),
            ([0, 1, .5, 0 ],       [0, 0, 0, 1, 0, 0 ], 'same', 0.3),
            ([0, 1, .5, 0, 0, 0 ], [0, 0, 0, 1, 0 ],    'same', 0.3),
            ([0, 1, .5, 0, 0, 0 ], [0, 0, 0, 1, ],      'same', 0.3),
            ([0, 1, .5],           [0, 0, 0, 1, 0, 0 ], 'valid', 0.3),
            ([0, 1, .5, 0],        [0, 0, 0, 1, 0, 0 ], 'valid', 0.3),
            ([0, 1, .5, 0, 0, 0 ], [0, 1, 0 ],          'valid', 0.1),
            ([0, 1, .5, 0, 0, 0 ], [0, 1, 0, 0 ],       'valid', 0.1),
            ([0, 1, .5],           [0, 0, 0, 1, 0, 0 ], 'full', 0.3),
            ([0, 1, .5, 0, 0, 0 ], [0, 1, 0 ],          'full', 0.1),
            ([0, 1, .5, 0],        [0, 0, 0, 1, 0, 0 ], 'full', 0.3),
            ([0, 1, .5, 0, 0, 0 ], [0, 1, 0, 0 ],       'full', 0.1)]:
           
            for ia in range(4):
                for ib in range(4):

                    ya = floats(la + [ 0 ] * ia)
                    yb = floats(lb + [ 0 ] * ib)

                    a = trace.Trace(tmin=10., deltat=0.1, ydata=ya)
                    b = trace.Trace(tmin=10.1, deltat=0.1, ydata=yb)
                    c = trace.correlate(a,b, mode=mode)
                   
                    if mode == 'valid' and c.ydata.size <=2:
                        continue

                    if not numeq(c.max(), [res, 1.], 0.0001):
                        print mode
                        print len(ya), ya
                        print len(yb), yb
                        print c.ydata, c.max()
                        assert False

    def testCorrelateNormalization(self):

        ya = floats([1,2,1])
        yb = floats([0,0,0,0,0,0,0,0,1,2,3,2,1])

        a = trace.Trace(tmin=sometime, deltat=0.1, ydata=ya)
        b = trace.Trace(tmin=sometime, deltat=0.1, ydata=yb)
        
        c_ab = trace.correlate(a,b)
        c_ab2 = trace.correlate(a,b, normalization='gliding')
        c_ba = trace.correlate(b,a)
        c_ba2 = trace.correlate(b,a, normalization='gliding')
            
        assert numeq( c_ab.ydata, c_ba.ydata[::-1], 0.001 )
        assert numeq( c_ab2.ydata, c_ba2.ydata[::-1], 0.001 )

    def testMovingSum(self):

        x = num.arange(5)
        assert numeq( trace.moving_sum(x,3,mode='valid'), [3,6,9], 0.001 )
        assert numeq( trace.moving_sum(x,3,mode='full'), [0,1,3,6,9,7,4], 0.001 )

    
    def testContinuousDownsample(self):

        y = num.random.random(1000)

        for (dt1, dt2) in [ (0.1, 1.0), (0.2, 1.0), (0.5, 1.0), (0.2, 0.4), (0.4, 1.2) ]:
            for tadd in (0.0, 0.01, 0.2, 0.5, 0.7, 0.75):
                a = trace.Trace(tmin=sometime+tadd, deltat=dt1, ydata=y)
                bs = [ trace.Trace(location='b', tmin=sometime+i*dt1*100+tadd, deltat=dt1, ydata=y[i*100:(i+1)*100]) for i in range(10) ]
                
                a.downsample_to(dt2,demean=False, snap=True)
                downsampler = trace.co_downsample_to(dt2)
                c2s = []
                for b in bs:
                    c = downsampler.send(b)
                    if c.data_len() > 0:
                        c2s.append(c)

                downsampler.close()
                assert  (round(c2s[0].tmin / dt2) * dt2 - c2s[0].tmin )/dt1 < 0.5001

if __name__ == "__main__":
    util.setup_logging('test_trace', 'warning')
    unittest.main()

