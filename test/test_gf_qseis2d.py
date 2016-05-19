import random
import math
import unittest
import logging
from tempfile import mkdtemp
import numpy as num

from pyrocko import util, trace, gf, cake  # noqa
from pyrocko.fomosto import qseis
import qseis2d

logger = logging.getLogger('test_gf_qseis2d')

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.
slowness_window = (0.0, 0.0, 0.4, 0.5)

class GFQSeis2dTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.tempdirs = []

    def __del__(self):
        import shutil

        for d in self.tempdirs:
            shutil.rmtree(d)

    def test_pyrocko_gf_vs_qseis2d(self):

        mod = cake.LayeredModel.from_scanlines(cake.read_nd_model_str('''
 0. 5.8 3.46 2.6 1264. 600.
 20. 5.8 3.46 2.6 1264. 600.
 20. 6.5 3.85 2.9 1283. 600.
 35. 6.5 3.85 2.9 1283. 600.
mantle
 35. 8.04 4.48 3.58 1449. 600.
 77.5 8.045 4.49 3.5 1445. 600.
 77.5 8.045 4.49 3.5 180.6 75.
 120. 8.05 4.5 3.427 180. 75.
 120. 8.05 4.5 3.427 182.6 76.06
 165. 8.175 4.509 3.371 188.7 76.55
 210. 8.301 4.518 3.324 201. 79.4
 210. 8.3 4.52 3.321 336.9 133.3
 410. 9.03 4.871 3.504 376.5 146.1
 410. 9.36 5.08 3.929 414.1 162.7
 660. 10.2 5.611 3.918 428.5 172.9
 660. 10.79 5.965 4.229 1349. 549.6'''.lstrip()))

        receiver_mod = cake.LayeredModel.from_scanlines(
                                    cake.read_nd_model_str('''
 0. 5.8 3.46 2.6 1264. 600.
 20. 5.8 3.46 2.6 1264. 600.
 20. 6.5 3.85 2.9 1283. 600.
 35. 6.5 3.85 2.9 1283. 600.
mantle
 35. 8.04 4.48 3.58 1449. 600.
'''.lstrip()))

        store_dir = mkdtemp(prefix='gfstore')
        self.tempdirs.append(store_dir)

        qconf = qseis2d.QSeis2dConfigFull()
        qconf.qseiss_version = '2014'
        qconf.qseiss_outdir = store_dir + '/' + qseis2d.default_qseiss_outdir
        print qconf.qseiss_outdir
        qconf.qseisr_version = '2014'

        qconf.calc_slowness_window = 0
        qconf.slowness_window = slowness_window

        qconf.time_region = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qconf.cut = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qconf.sw_flat_earth_transform = 0

        config = gf.meta.ConfigTypeA(
            id='qseis2d_test',
            ncomponents=10,
            sample_rate=0.25,
            receiver_depth=0.*km,
            source_depth_min=10*km,
            source_depth_max=10*km,
            source_depth_delta=1*km,
            distance_min=550*km,
            distance_max=560*km,
            distance_delta=1*km,
            modelling_code_id='qseis2d',
            earthmodel_1d=mod,
            earthmodel_receiver_1d = receiver_mod,
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
            store_dir, config=config, extra={'qseis2d': qconf})

        store = gf.store.Store(store_dir, 'r')
        store.make_ttt()
        store.close()

        try:
            qseis2d.build(store_dir, nworkers=1)
        except qseis2d.QSeis2dError, e:
            if str(e).find('could not start qseis2d') != -1:
                logger.warn('qseis2d not installed; '
                            'skipping test_pyrocko_gf_vs_qseis2d')
                return
            else:
                raise

        source = gf.MTSource(
            lat=0.,
            lon=0.,
            depth=10.*km)

        source.m6 = tuple(random.random()*2.-1. for x in xrange(6))

        azi = 0.    # QSeis2d only takes one receiver without azimuth variable
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
                store_id='qseis2d_test')

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

        runner = qseis.QSeisRunner(keep_tmp=True)
        conf = qseis.QSeisConfigFull.example()
        conf.qseis_version = '2006a'
        conf.wavelet_type = 1
        conf.wavelet_duration_samples = 0.25
        conf.receiver_distances = [dist/km]
        conf.receiver_azimuths = [azi]
        conf.source_depth = source.depth/km
        conf.aliasing_suppression_factor = 0.1
        conf.sw_algorithm = 1
        conf.slowness_window = slowness_window
        conf.time_window = 508.
        conf.nsamples = 128
        conf.sw_flat_earth_transform = 0
        conf.source_mech = qseis.QSeisSourceMechMT(
            mnn=source.mnn,
            mee=source.mee,
            mdd=source.mdd,
            mne=source.mne,
            mnd=source.mnd,
            med=source.med)
        conf.earthmodel_1d = mod
        conf.time_reduction_velocity = 15.0
        conf.time_start = 0.0

        runner.run(conf)

        trs1 = runner.get_traces()

        # stf of Qseis2d has to be the same
        wavelet_duration = 2 * (1 / config.sample_rate)

        runner2d = qseis2d.QSeisRRunner(tmp=store_dir, keep_tmp=True)
        conf2d = qseis2d.QSeis2dConfigFull()
        conf2d.qseiss_outdir = qconf.qseiss_outdir
        conf2d.qseisr_version = '2014'
        conf2d.qseiss_version = '2014'
        conf2d.receiver = qseis2d.QSeisRReceiver(lat=90 - dist * cake.m2d,
                                           lon=0.0,
                                           tstart=0.0,
                                           distance=dist)
        conf2d.source = qseis2d.QSeis2dSource(lat=90,
                                        lon=0.0,
                                        depth=source.depth/km)
        conf2d.wavelet_duration = wavelet_duration
        conf.time_reduction = 0.
        conf2d.time_window = 508.
        conf2d.nsamples = 128
        conf2d.source_mech = qseis2d.QSeisRSourceMechMT(
            mnn=source.mnn,
            mee=source.mee,
            mdd=source.mdd,
            mne=source.mne,
            mnd=source.mnd,
            med=source.med)
        conf2d.earthmodel_1d = mod
        conf2d.earthmodel_receiver_1d = receiver_mod
        runner2d.run(conf2d)

        trs2 = runner2d.get_traces()

        engine = gf.LocalEngine(store_dirs=[store_dir])
        trs3 = engine.process(source, targets).pyrocko_traces()

        for tr in trs1:
            tr.station = 'QS'

        for tr in trs2:
            tr.station = 'QS2'

        for tr in trs3:
            tr.station = 'GFQ2'

        trace.snuffle(trs1+trs2+trs3)

if __name__ == '__main__':
    util.setup_logging('test_gf_qseis', 'warning')
    unittest.main()
