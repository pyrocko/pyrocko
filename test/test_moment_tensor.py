import unittest
import random, math
import numpy as num

from pyrocko.moment_tensor import *


class MomentTensorTestCase( unittest.TestCase ):
    
    def testMagnitudeMoment(self):
        for i in range(1,10):
            mag = float(i)
            assert abs(mag - moment_to_magnitude(magnitude_to_moment(mag))) < 1e-6, \
                'Magnitude to moment to magnitude test failed.'
            
    def testAnyAngles(self):
        '''Check some arbitrary angles.'''
        for i in range(100):
            (s1,d1,r1) = [ r2d*random.random()*10.-5. for j in range(3) ]
            m0 = 1.+random.random()*1.0e20
            self.forwardBackward( s1, d1, r1, m0 )

    def testProblematicAngles(self):
        '''Checks angles close to fractions of pi, which are especially problematic.'''
        for i in range(100):
            # angles close to fractions of pi are especially problematic
            (s1,d1,r1) = [ r2d*random.randint(-16,16)*math.pi/8.+1e-8*random.random()-0.5e-8 for j in range(3) ]
            m0 = 1.+random.random()*1.0e20
            self.forwardBackward( s1, d1, r1, m0 )
                
    def testNonPlainDoubleCouples(self):
        '''Convert random MTs to plain double couples and compare angles.'''
        for i in range(100):
            ms = [ random.random()*1.0e20-0.5e20 for j in range(6) ]
            m = num.matrix([[ ms[0],ms[3],ms[4]],[ms[3],ms[1],ms[5]],[ms[4],ms[5],ms[2]]], dtype=num.float )
            
            m1 = MomentTensor( m=m )
            m_plain = m1.m_plain_double_couple()
            m2 = MomentTensor( m=m_plain )
            
            self.assertAnglesSame( m1, m2 )           
                
    def forwardBackward(self, strike, dip, rake, scalar_moment ):
        m1 = MomentTensor( strike=strike, dip=dip, rake=rake, scalar_moment=scalar_moment )
        m2 = MomentTensor( m=m1.m() )
        self.assertAnglesSame(m1,m2)
        
    def assertAnglesSame(self, m1, m2):
        assert num.all( num.abs(num.array(m1.both_strike_dip_rake()) - 
                                num.array(m2.both_strike_dip_rake())) < 1e-7*100 ), \
            "angles don't match after forward-backward calculation:\nfirst:\n"+str(m1)+ "\nsecond:\n"+str(m2)

if __name__ == "__main__":
    pyrocko.util.setup_logging('warning')
    unittest.main()
