from pyrocko import trace, io, util
import unittest
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
        
        
if __name__ == "__main__":
    util.setup_logging('warning')
    unittest.main()

