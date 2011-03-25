from pyrocko import mseed, trace, util, io
import unittest, math, calendar, time
from random import random

class UtilTestCase( unittest.TestCase ):
    
    def testTime(self):
        
        for fmt, accu in zip(
            [ '%Y-%m-%d %H:%M:%S.3FRAC', '%Y-%m-%d %H:%M:%S.2FRAC', '%Y-%m-%d %H:%M:%S.1FRAC', '%Y-%m-%d %H:%M:%S' ],
            [ 0.001, 0.01, 0.1, 1.] ):
        
            ta = util.str_to_time('1960-01-01 10:10:10')
            tb = util.str_to_time('2020-01-01 10:10:10')
            
            for i in xrange(10000):
                t1 = ta + random() * (tb-ta)
                s = util.time_to_str(t1, format=fmt)
                t2 = util.str_to_time(s, format=fmt)
                assert abs( t1 - t2 ) < accu
    
        

if __name__ == "__main__":
    util.setup_logging('test_util', 'warning')
    unittest.main()
    