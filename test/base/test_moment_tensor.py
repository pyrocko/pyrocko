import unittest
import random
import math

import numpy as num

from pyrocko.moment_tensor import \
    magnitude_to_moment, moment_to_magnitude, MomentTensor, r2d, symmat6, \
    dynecm, kagan_angle, rotation_from_angle_and_axis, random_axis

from pyrocko import util, guts


class MomentTensorTestCase(unittest.TestCase):

    def testMagnitudeMoment(self):
        for i in range(1, 10):
            mag = float(i)
            assert abs(mag - moment_to_magnitude(magnitude_to_moment(mag))) \
                < 1e-6, 'Magnitude to moment to magnitude test failed.'

    def testAnyAngles(self):
        for i in range(100):
            (s1, d1, r1) = [r2d*random.random()*10.-5. for j in range(3)]
            m0 = 1.+random.random()*1.0e20
            self.forwardBackward(s1, d1, r1, m0)

    def testProblematicAngles(self):
        for i in range(100):
            # angles close to fractions of pi are especially problematic
            (s1, d1, r1) = [
                r2d*random.randint(-16, 16)*math.pi/8. +
                1e-8*random.random()-0.5e-8 for j in range(3)]
            m0 = 1.+random.random()*1.0e20
            self.forwardBackward(s1, d1, r1, m0)

    def testNonPlainDoubleCouples(self):
        for i in range(100):
            ms = [random.random()*1.0e20-0.5e20 for j in range(6)]
            m = num.array(
                [[ms[0], ms[3], ms[4]],
                 [ms[3], ms[1], ms[5]],
                 [ms[4], ms[5], ms[2]]], dtype=float)

            m1 = MomentTensor(m=m)
            m_plain = m1.m_plain_double_couple()
            m2 = MomentTensor(m=m_plain)

            self.assertAnglesSame(m1, m2)

    def forwardBackward(self, strike, dip, rake, scalar_moment):
        m1 = MomentTensor(
            strike=strike, dip=dip, rake=rake, scalar_moment=scalar_moment)
        m2 = MomentTensor(m=m1.m())
        self.assertAnglesSame(m1, m2)

    def assertAnglesSame(self, m1, m2):
        assert num.all(num.abs(num.array(m1.both_strike_dip_rake()) -
                               num.array(m2.both_strike_dip_rake()))
                       < 1e-7*100), \
            "angles don't match after forward-backward calculation:\n' \
            'first:\n"+str(m1) + "\nsecond:\n"+str(m2)

    def assertSame(self, a, b, eps, errstr):
        assert num.all(num.abs(num.array(a)-num.array(b)) < eps), errstr

    def testChile(self):
        m_use = symmat6(
            1.040, -0.030, -1.010, 0.227, -1.510, -0.120)*1e29*dynecm
        mt = MomentTensor(m_up_south_east=m_use)
        sdr = mt.both_strike_dip_rake()
        self.assertSame(sdr[0], (174., 73., 83.), 1., 'chile fail 1')
        self.assertSame(sdr[1], (18., 18., 112.), 1., 'chile fail 2')

    def testENU(self):
        m_enu = num.array(
            [[0.66, 0.53, -0.18],
             [0.53, -0.70, -0.35],
             [-0.18, -0.36, 0.04]], dtype=float)
        m = MomentTensor(m_east_north_up=m_enu)
        self.assertEqual(m_enu[0, 0], m.m()[1, 1])
        self.assertEqual(m_enu[1, 1], m.m()[0, 0])
        self.assertEqual(m_enu[2, 2], m.m()[2, 2])
        self.assertEqual(m_enu[0, 1], m.m()[0, 1])
        self.assertEqual(m_enu[0, 2], -m.m()[1, 2])
        self.assertEqual(m_enu[1, 2], -m.m()[0, 2])
        self.assertEqual(m_enu[0, 0], m.m6_east_north_up()[0])
        self.assertEqual(m_enu[1, 1], m.m6_east_north_up()[1])
        self.assertEqual(m_enu[2, 2], m.m6_east_north_up()[2])
        self.assertEqual(m_enu[0, 1], m.m6_east_north_up()[3])
        self.assertEqual(m_enu[0, 2], m.m6_east_north_up()[4])
        self.assertEqual(m_enu[1, 2], m.m6_east_north_up()[5])
        self.assertSame(m_enu, m.m_east_north_up(), 1e-3, 'testENU fail')

    def testIO(self):
        m1 = MomentTensor(dip=90.)
        sdr1 = m1.both_strike_dip_rake()
        m2 = guts.load(string=m1.dump())
        sdr2 = m2.both_strike_dip_rake()
        self.assertSame(sdr1, sdr2, 0.1, 'failed io via guts')

    def testProps(self):
        m = MomentTensor()
        m.mnn = 1.
        m.mee = -1.
        m.mdd = 0.
        m.mne = 0.
        m.mnd = 0.
        m.med = 0.
        (s1, d1, _), (s2, d2, _) = m.both_strike_dip_rake()
        assert abs(s1 - 45.) < 0.1 or abs(s2 - 45.) < 0.1

    def testMomentGetterSetter(self):
        m1 = MomentTensor()
        want_mom = 2E7
        m1.moment = want_mom
        sm2 = m1.scalar_moment()
        assert sm2 == m1.moment
        assert abs(sm2 - want_mom) < 1E-8

        mag = moment_to_magnitude(want_mom)
        assert m1.magnitude == mag

        want_mag = 3.5
        m1.magnitude = want_mag
        mom = magnitude_to_moment(want_mag)
        assert m1.moment == mom

    def testKagan(self):
        eps = 0.01
        for _ in range(500):
            mt1 = MomentTensor.random_mt(magnitude=-1.0)
            assert 0.0 == kagan_angle(mt1, mt1)

            mt1 = MomentTensor.random_dc(magnitude=-1.0)
            assert 0.0 == kagan_angle(mt1, mt1)

            angle = random.random() * 90.

            rot = rotation_from_angle_and_axis(angle, random_axis())

            mrot = num.dot(rot.T, num.dot(mt1.m(), rot))
            mt2 = MomentTensor(m=mrot)
            angle2 = kagan_angle(mt1, mt2)

            assert abs(angle - angle2) < eps

    def testRandomRotation(self):
        angles = []
        rstate = num.random.RandomState(10)
        for _ in range(100):
            mt1 = MomentTensor.random_mt(magnitude=-1.0)
            mt2 = mt1.random_rotated(10.0, rstate=rstate)
            angle = kagan_angle(mt1, mt2)
            angles.append(angle)

        assert abs(num.mean(angles) - 8.330) < 0.01

        mt1 = MomentTensor.random_mt(magnitude=-1.0)
        mt2 = mt1.random_rotated(angle=10.0)

        assert abs(kagan_angle(mt1, mt2) - 10.0) < 0.0001

    def test_pt_to_sdr(self):
        for _ in range(100):
            mt1 = MomentTensor.random_dc(scalar_moment=1.0)
            sdr1, sdr2 = mt1.both_strike_dip_rake()
            p_axis = mt1.p_axis()
            t_axis = mt1.t_axis()
            mt2 = MomentTensor(p_axis=p_axis, t_axis=t_axis)
            assert num.all((mt1.m() - mt2.m()) < 1e-6)


if __name__ == "__main__":
    util.setup_logging('test_moment_tensor', 'warning')
    unittest.main()
