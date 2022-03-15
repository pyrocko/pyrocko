from __future__ import division, print_function, absolute_import

import unittest
import logging
from tempfile import mkdtemp
import shutil

from pyrocko import util, trace, gf, cake  # noqa
from pyrocko.fomosto import qseis
from pyrocko.fomosto.report import GreensFunctionTest as gftest

logger = logging.getLogger('pyrocko.test.test_fomosto_report')

km = 1e3


class ReportTestCase(unittest.TestCase):
    tempdirs = []

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        for d in cls.tempdirs:
            shutil.rmtree(d)

    def test_pyrocko_report_with_gf_and_qseis(self):

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

        store_dir = mkdtemp(prefix='gft_')
        self.tempdirs.append(store_dir)

        qsconf = qseis.QSeisConfig()
        qsconf.qseis_version = '2006a'

        qsconf.time_region = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qsconf.cut = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qsconf.wavelet_duration_samples = 0.001
        qsconf.sw_flat_earth_transform = 0

        config = gf.meta.ConfigTypeA(
            id='gft_test',
            ncomponents=10,
            sample_rate=0.25,
            receiver_depth=0.*km,
            source_depth_min=10*km,
            source_depth_max=10*km,
            source_depth_delta=1*km,
            distance_min=550*km,
            distance_max=560*km,
            distance_delta=1*km,
            modelling_code_id='qseis.2006a',
            earthmodel_1d=mod,
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
        store.make_travel_time_tables()
        store.close()

        try:
            qseis.build(store_dir, nworkers=1)
        except qseis.QSeisError as e:
            if str(e).find('could not start qseis') != -1:
                logger.warning('qseis not installed; '
                               'skipping test_pyrocko_gf_vs_qseis')
                return
            else:
                raise

        gft = gftest(store_dir, sensor_count=2, pdf_dir=store_dir,
                     plot_velocity=True, rel_lowpass_frequency=(1. / 110),
                     rel_highpass_frequency=(1. / 16), output_format='html')

        src = gft.createSource('DC', None, 45., 90., 180.)
        sen = gft.createSensors(strike=0., codes=('', 'STA', '', 'Z'),
                                azimuth=0., dip=-90.)
        gft.trace_configs = [[src, sen]]
        gft.createDisplacementTraces()
        gft.createVelocityTraces()
        gft.applyFrequencyFilters()
        gft.getPhaseArrivals()
        gft.createOutputDoc()


if __name__ == '__main__':
    util.setup_logging('test_fomosto_report', 'warning')
    unittest.main()
