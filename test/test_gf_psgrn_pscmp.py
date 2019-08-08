from __future__ import division, print_function, absolute_import
import math
from time import time
import unittest
import logging
from tempfile import mkdtemp
import numpy as num
import os
import shutil

from pyrocko import orthodrome as ortd
from pyrocko import util, gf, cake  # noqa
from pyrocko.fomosto import psgrn_pscmp
from .common import Benchmark

logger = logging.getLogger('pyrocko.test.test_gf_psgrn_pscmp')
benchmark = Benchmark()

km = 1000.

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.


def statics(engine, source, starget):
    store = engine.get_store(starget.store_id)
    dsource = source.discretize_basesource(store, starget)

    assert len(starget.north_shifts) == len(starget.east_shifts)

    out = num.zeros((len(starget.north_shifts), len(starget.components)))
    sfactor = source.get_factor()
    for i, (north, east) in enumerate(
            zip(starget.north_shifts, starget.east_shifts)):

        receiver = gf.Receiver(
            lat=starget.lat,
            lon=starget.lon,
            north_shift=north,
            east_shift=east,
            depth=starget.depth)

        values = store.statics(
            dsource, receiver, starget.components,
            interpolation=starget.interpolation)

        for icomponent, value in enumerate(values):
            out[i, icomponent] = value * sfactor

    return out


@unittest.skipUnless(
    psgrn_pscmp.have_backend(), 'backend psgrn_pscmp not available')
class GFPsgrnPscmpTestCase(unittest.TestCase):

    tempdirs = []

    def __init__(self, *args, **kwargs):
        self.pscmp_store_dir = None
        unittest.TestCase.__init__(self, *args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        for d in cls.tempdirs:
            shutil.rmtree(d)

    def get_pscmp_store_info(self):
        if self.pscmp_store_dir is None:
            self.pscmp_store_dir, self.psgrn_config = self._create_psgrn_pscmp_store()

        return self.pscmp_store_dir, self.psgrn_config

    def _create_psgrn_pscmp_store(self):

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

        store_dir = mkdtemp(prefix='gfstore')
        self.tempdirs.append(store_dir)
        store_id = 'psgrn_pscmp_test'

        version = '2008a'

        c = psgrn_pscmp.PsGrnPsCmpConfig()
        c.psgrn_config.sampling_interval = 1.
        c.psgrn_config.version = version
        c.pscmp_config.version = version

        config = gf.meta.ConfigTypeA(
            id=store_id,
            ncomponents=10,
            sample_rate=1. / (3600. * 24.),
            receiver_depth=0. * km,
            source_depth_min=0. * km,
            source_depth_max=5. * km,
            source_depth_delta=0.1 * km,
            distance_min=0. * km,
            distance_max=40. * km,
            distance_delta=0.1 * km,
            modelling_code_id='psgrn_pscmp.%s' % version,
            earthmodel_1d=mod,
            tabulated_phases=[])

        config.validate()
        gf.store.Store.create_editables(
            store_dir, config=config, extra={'psgrn_pscmp': c})

        store = gf.store.Store(store_dir, 'r')
        store.close()

        # build store
        try:
            psgrn_pscmp.build(store_dir, nworkers=1)
        except psgrn_pscmp.PsCmpError as e:
            if str(e).find('could not start psgrn/pscmp') != -1:
                logger.warn('psgrn/pscmp not installed; '
                            'skipping test_pyrocko_gf_vs_pscmp')
                return
            else:
                raise

        return store_dir, c

    def test_fomosto_vs_psgrn_pscmp(self):

        store_dir, c = self.get_pscmp_store_info()

        origin = gf.Location(
            lat=10.,
            lon=-15.)

        # test GF store
        TestRF = dict(
            lat=origin.lat,
            lon=origin.lon,
            depth=2. * km,
            width=0.2 * km,
            length=0.5 * km,
            rake=90., dip=45., strike=45.,
            slip=1.)

        source_plain = gf.RectangularSource(**TestRF)
        source_with_time = gf.RectangularSource(time=123.5, **TestRF)

        neast = 40
        nnorth = 40

        N, E = num.meshgrid(num.linspace(-20. * km, 20. * km, nnorth),
                            num.linspace(-20. * km, 20. * km, neast))

        # direct pscmp output
        lats, lons = ortd.ne_to_latlon(
            origin.lat, origin.lon, N.flatten(), E.flatten())
        pscmp_sources = [psgrn_pscmp.PsCmpRectangularSource(**TestRF)]

        cc = c.pscmp_config
        cc.observation = psgrn_pscmp.PsCmpScatter(lats=lats, lons=lons)

        cc.rectangular_source_patches = pscmp_sources

        ccf = psgrn_pscmp.PsCmpConfigFull(**cc.items())
        ccf.psgrn_outdir = os.path.join(store_dir, c.gf_outdir) + '/'

        t2 = time()
        runner = psgrn_pscmp.PsCmpRunner(keep_tmp=False)
        runner.run(ccf)
        ps2du = runner.get_results(component='displ')[0]
        t3 = time()
        logger.info('pscmp stacking time %f' % (t3 - t2))

        un_pscmp = ps2du[:, 0]
        ue_pscmp = ps2du[:, 1]
        ud_pscmp = ps2du[:, 2]

        # test against engine
        starget = gf.StaticTarget(
            lats=num.array([origin.lat] * N.size),
            lons=num.array([origin.lon] * N.size),
            north_shifts=N.flatten(),
            east_shifts=E.flatten(),
            interpolation='nearest_neighbor')

        engine = gf.LocalEngine(store_dirs=[store_dir])

        for source in [source_plain, source_with_time]:
            t0 = time()
            r = engine.process(source, starget)
            t1 = time()
            logger.info('pyrocko stacking time %f' % (t1 - t0))
            un_fomosto = r.static_results()[0].result['displacement.n']
            ue_fomosto = r.static_results()[0].result['displacement.e']
            ud_fomosto = r.static_results()[0].result['displacement.d']

            num.testing.assert_allclose(un_fomosto, un_pscmp, atol=0.002)
            num.testing.assert_allclose(ue_fomosto, ue_pscmp, atol=0.002)
            num.testing.assert_allclose(ud_fomosto, ud_pscmp, atol=0.002)


if __name__ == '__main__':
    util.setup_logging('test_gf_psgrn_pscmp', 'warning')
    unittest.main()
