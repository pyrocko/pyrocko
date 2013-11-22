import unittest, os
import numpy as num
from pyrocko import util, evalresp, pz, trace

def numeq(a,b, eps):
    return num.all(num.abs(num.array(a) - num.array(b)) < eps)

class ResponseTestCase(unittest.TestCase):
    
    def test_evalresp(self, plot=False):

        testdir = os.path.dirname(__file__)

        freqs = num.logspace(num.log10(0.001), num.log10(10.), num=1000)

        transfer = evalresp.evalresp(sta_list='BSEG',
                          cha_list='BHZ',
                          net_code='GR',
                          locid='',
                          instant=util.str_to_time('2012-01-01 00:00:00'),
                          freqs=freqs,
                          units='DIS',
                          file=os.path.join(testdir, 'response', 'RESP.GR.BSEG..BHZ'),
                          rtype='CS')[0][4]

        pzfn = 'SAC_PZs_GR_BSEG_BHZ__2008.254.00.00.00.0000_2599.365.23.59.59.99999'

        zeros, poles, constant = pz.read_sac_zpk(filename=os.path.join(
            testdir, 'response', pzfn))
        
        resp = trace.PoleZeroResponse(zeros, poles, constant)

        transfer2 = resp.evaluate(freqs)

        if plot:
            import pylab as lab
            lab.plot(freqs, num.imag(transfer))
            lab.plot(freqs, num.imag(transfer2))
            lab.gca().loglog() 
            lab.show()

        assert num.max(num.abs(transfer - transfer2)) < 1e-4

if __name__ == "__main__":
    util.setup_logging('test_response', 'warning')
    unittest.main()
