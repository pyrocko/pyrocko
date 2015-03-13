import random
import math
import unittest
import logging
from tempfile import mkdtemp
import numpy as num

from pyrocko import util, trace, gf, cake  # noqa
from pyrocko.fomosto import qseis

logger = logging.getLogger('test_gf_qseis')

km = 1000.

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.


class GFQSeisTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.tempdirs = []

    def __del__(self):
        import shutil

        for d in self.tempdirs:
            shutil.rmtree(d)

    def test_pyrocko_gf_vs_qseis(self):

        store_dir = mkdtemp(prefix='gfstore')
        self.tempdirs.append(store_dir)

        qsconf = qseis.QSeisConfig()

        qsconf.time_region = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qsconf.cut = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qsconf.wavelet_duration_samples = 0.001
        qsconf.sw_flat_earth_transform = 0

        config = gf.meta.ConfigTypeA(
            id='qseis_test',
            ncomponents=10,
            sample_rate=0.25,
            receiver_depth=0.*km,
            source_depth_min=10*km,
            source_depth_max=10*km,
            source_depth_delta=1*km,
            distance_min=550*km,
            distance_max=560*km,
            distance_delta=1*km,
            modelling_code_id='qseis',
            earthmodel_1d=cake.load_model('mod.nd'),
            tabulated_phases=[
                gf.meta.TPDef(
                    id='begin',
                    definition='p,P,p\\,P\\'),
                gf.meta.TPDef(
                    id='end',
                    definition='2.5'),
            ])

        config.validate()
        gf.store.Store.create_editables(
            store_dir, config=config, extra={'qseis': qsconf})

        store = gf.store.Store(store_dir, 'r')
        store.make_ttt()
        store.close()

        try:
            qseis.build(store_dir, nworkers=1)
        except qseis.QSeisError, e:
            if str(e).find('could not start qseis') != -1:
                logger.warn('qseis not installed; '
                            'skipping test_pyrocko_gf_vs_qseis')
                return
            else:
                raise

        source = gf.MTSource(
            lat=0.,
            lon=0.,
            depth=10.*km)

        source.m6 = tuple(random.random()*2.-1. for x in xrange(6))

        azi = random.random()*365.
        dist = 553.*km

        dnorth = dist * math.cos(azi*d2r)
        deast = dist * math.sin(azi*d2r)

        targets = []
        for cha in 'rtz':
            target = gf.Target(
                quantity='displacement',
                codes=('', '0000', 'PG', cha),
                north_shift=dnorth,
                east_shift=deast,
                store_id='qseis_test')

            dist = source.distance_to(target)
            azi, bazi = source.azibazi_to(target)

            if cha == 'r':
                target.azimuth = bazi + 180.
                target.dip = 0.
            elif cha == 't':
                target.azimuth = bazi - 90.
                target.dip = 0.
            elif cha == 'z':
                target.azimuth = 0.
                target.dip = 90.

            targets.append(target)

        runner = qseis.QSeisRunner()
        conf = qseis.QSeisConfigFull.example()
        conf.receiver_distances = [dist/km]
        conf.receiver_azimuths = [azi]
        conf.source_depth = source.depth/km
        conf.time_window = 508.
        conf.nsamples = 128
        conf.source_mech = qseis.QSeisSourceMechMT(
            mnn=source.mnn,
            mee=source.mee,
            mdd=source.mdd,
            mne=source.mne,
            mnd=source.mnd,
            med=source.med)
        conf.earthmodel_1d = cake.load_model('mod.nd')

        runner.run(conf)

        trs = runner.get_traces()
        for tr in trs:
            tr.shift(-3.)
            tr.snap(interpolate=True)
            tr.lowpass(4, 0.05)
            tr.highpass(4, 0.01)

        engine = gf.LocalEngine(store_dirs=[store_dir])
        trs2 = engine.process(source, targets).pyrocko_traces()
        for tr in trs2:
            tr.snap(interpolate=True)
            tr.lowpass(4, 0.05)
            tr.highpass(4, 0.01)

        def g(trs, cha):
            for tr in trs:
                if tr.channel == cha:
                    return tr

        for cha in 'rtz':
            t1 = g(trs, cha)
            t2 = g(trs2, cha)
            tmin = max(t1.tmin, t2.tmin)
            tmax = min(t1.tmax, t2.tmax)
            t1.chop(tmin, tmax)
            t2.chop(tmin, tmax)
            d = 2.0 * num.sum((t1.ydata - t2.ydata)**2) / \
                (num.sum(t1.ydata**2) + num.sum(t2.ydata**2))

            assert d < 0.05

        # trace.snuffle(trs+trs2)

if __name__ == '__main__':
    util.setup_logging('test_gf_qseis', 'warning')
    unittest.main()
