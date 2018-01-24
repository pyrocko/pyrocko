from __future__ import division, print_function, absolute_import
import sys
import math
import unittest
import numpy as num
from tempfile import mkdtemp
import shutil

from pyrocko import gf, util, guts, cake

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.


def numeq(a, b, eps):
    return (num.all(num.asarray(a).shape == num.asarray(b).shape and
            num.abs(num.asarray(a) - num.asarray(b)) < eps))


class GFSourcesTestCase(unittest.TestCase):
    tempdirs = []

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self._dummy_store = None

    @classmethod
    def tearDownClass(cls):
        for d in cls.tempdirs:
            shutil.rmtree(d)

    if sys.version_info < (2, 7):
        from contextlib import contextmanager

        @contextmanager
        def assertRaises(self, exc):

            gotit = False
            try:
                yield None
            except exc:
                gotit = True

            assert gotit, 'expected to get a %s exception' % exc

        def assertIsNone(self, value):
            assert value is None, 'expected None but got %s' % value

    def test_source_to_event(self):

        for S in gf.source_classes:
            stf = gf.TriangularSTF(effective_duration=2.0)
            s1 = S(lat=10., lon=20., depth=1000.,
                   north_shift=500., east_shift=500., stf=stf)
            ev = s1.pyrocko_event()
            s2 = S.from_pyrocko_event(ev)

            if not isinstance(s1, gf.DoubleDCSource):
                assert numeq(
                    [s1.effective_lat, s1.effective_lon,
                     s1.depth, s1.stf.effective_duration],
                    [s2.effective_lat, s2.effective_lon,
                     s2.depth, s2.stf.effective_duration], 0.001)
            else:
                assert numeq(
                    [s1.effective_lat, s1.effective_lon,
                     s1.depth, s1.stf.effective_duration],
                    [s2.effective_lat, s2.effective_lon,
                     s2.depth, s2.stf1.effective_duration], 0.001)

    def test_source_dict(self):
        s1 = gf.DCSource(strike=0.)
        s1.strike = 10.
        s2 = s1.clone(strike=20.)
        s2.update(strike=30.)
        s2['strike'] = 40.
        d = dict(s2)
        s3 = gf.DCSource(**d)
        s3.strike

    def test_sgrid(self):

        r = gf.Range

        source = gf.DCSource()
        sgrid = source.grid(rake=r(-10, 10, 1),
                            strike=r(-100, 100, n=21),
                            depth=r('0k .. 100k : 10k'),
                            magnitude=r(1, 2, 1))

        sgrid = guts.load_string(sgrid.dump())

        n = len(sgrid)
        i = 0
        for source in sgrid:
            i += 1

        assert i == n

    def test_sgrid2(self):
        expect = [10., 12., 14., 16., 18., 20.]
        source = gf.DCSource()
        sgrid = source.grid(dip=gf.Range(10, 20, 2))
        dips = []
        for source in sgrid:
            dips.append(source.dip)

        num.testing.assert_array_almost_equal(
            dips, expect)

        source = gf.DCSource(dip=10)
        sgrid = source.grid(dip=gf.Range(1, 2, 0.2, relative='mult'))
        dips = []
        for source in sgrid:
            dips.append(source.dip)

        num.testing.assert_array_almost_equal(
            dips, expect)

    def dummy_store(self):
        if self._dummy_store is None:

            conf = gf.ConfigTypeA(
                id='empty_regional',
                source_depth_min=0.,
                source_depth_max=20*km,
                source_depth_delta=10*km,
                distance_min=1000*km,
                distance_max=2000*km,
                distance_delta=10*km,
                sample_rate=2.0,
                ncomponents=10,
                earthmodel_1d=cake.load_model(crust2_profile=(50., 10.)))

            store_dir = mkdtemp(prefix='gfstore')
            self.tempdirs.append(store_dir)

            gf.Store.create(store_dir, config=conf)
            self._dummy_store = gf.Store(store_dir)

        return self._dummy_store

    def test_combine_dsources(self):
        store = self.dummy_store()
        dummy_target = gf.Target()
        for S in gf.source_classes:
            if not hasattr(S, 'discretize_basesource'):
                continue

            for lats in [[10., 10., 10.], [10., 11., 12.]]:
                sources = [
                    S(lat=lat, lon=20., depth=1000.,
                      north_shift=500., east_shift=500.)
                    for lat in lats]

                dsources = [
                    s.discretize_basesource(store, target=dummy_target)
                    for s in sources]

                DS = dsources[0].__class__

                dsource = DS.combine(dsources)
                assert dsource.nelements == sum(s.nelements for s in dsources)

    def test_source_times(self):
        store = self.dummy_store()
        dummy_target = gf.Target()
        for S in gf.source_classes:
            if not hasattr(S, 'discretize_basesource'):
                continue

            for t in [0.0, util.str_to_time('2014-01-01 10:00:00')]:
                source = S(time=t)
                dsource = source.discretize_basesource(
                    store, target=dummy_target)
                cent = dsource.centroid()
                assert numeq(cent.time + source.get_timeshift(), t, 0.0001)

    def test_outline(self):
        s = gf.MTSource(
            east_shift=5. * km,
            north_shift=-3. * km,
            depth=7. * km)

        outline = s.outline()
        numeq(outline, num.array([[-3000., 5000., 7000.]]), 1e-8)

        rs = gf.RectangularSource(
            length=2 * km,
            width=2 * km)

        outline = rs.outline()
        numeq(
            outline,
            num.array(
                [[-1.e3, 0.0, 0.0],
                 [1.e3, 0.0, 0.0],
                 [1.e3, 0.0, 2.e3],
                 [-1.e3, 0.0, 2.e3],
                 [-1.e3, 0.0, 0.0]]),
            1e-8)

    def test_rect_source_anchors(self):
        sources = {}
        for anchor in ['top', 'center', 'bottom']:
            sources[anchor] = gf.RectangularSource(
                width=5 * km,
                length=20 * km,
                anchor=anchor,
                dip=120.,
                strike=45.,
                east_shift=0 * km,
                north_shift=0 * km)

        def plot_sources(sources):
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.gca()
            colors = ['b', 'r', 'y']
            for i, src in enumerate(sources.itervalues()):
                n, e = src.outline(cs='xy').T
                ax.fill(e, n, color=colors[i], alpha=0.5)
            plt.show()

        # plot_sources(sources)

    def test_explosion_source(self):
        ex = gf.ExplosionSource(
                magnitude=5.,
                volume_change=4.,
                depth=5*km)

        with self.assertRaises(gf.DerivedMagnitudeError):
            ex.validate()

        ex = gf.ExplosionSource(
                depth=5*km)

        ex.validate()

        self.assertEqual(ex.get_moment(), 1.0)

        ex = gf.ExplosionSource(
                magnitude=3.0,
                depth=5*km)

        store = self.dummy_store()

        with self.assertRaises(gf.DerivedMagnitudeError):
            ex.get_volume_change()

        volume_change = ex.get_volume_change(store)

        ex = gf.ExplosionSource(
                volume_change=volume_change,
                depth=5*km)

        self.assertAlmostEqual(ex.get_magnitude(store), 3.0)

        ex = gf.ExplosionSource(
                magnitude=3.0,
                depth=-5.)

        with self.assertRaises(gf.DerivedMagnitudeError):
            ex.get_volume_change(store)


if __name__ == '__main__':
    util.setup_logging('test_gf_sources', 'warning')
    unittest.main()
