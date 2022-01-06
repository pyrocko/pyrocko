from __future__ import division, print_function, absolute_import

import random
import math
import unittest
import logging
from tempfile import mkdtemp
import numpy as num
import copy
import os
import shutil

from pyrocko import util, trace, gf, cake  # noqa
from pyrocko.fomosto import qseis
from pyrocko.fomosto import qseis2d

logger = logging.getLogger('pyrocko.test.test_qseis_qseis2d')

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1e3
slowness_window = (0.0, 0.0, 0.4, 0.5)


@unittest.skipUnless(
    qseis2d.have_backend(), 'backend qseis2d not available')
@unittest.skipUnless(
    qseis.have_backend(), 'backend qseis not available')
class GFQSeis2dTestCase(unittest.TestCase):

    tempdirs = []

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        for d in cls.tempdirs:
            shutil.rmtree(d)

    def test_pyrocko_gf_vs_qseis2d(self):

        mod = cake.LayeredModel.from_scanlines(cake.read_nd_model_str('''
 0. 5.8 3.46 2.6 1264. 600.
 20. 5.8 3.46 2.6 1264. 600.
 20. 6.5 3.85 2.9 1283. 600.
 35. 6.5 3.85 2.9 1283. 600.
mantle
 35. 8.04 4.48 3.58 1449. 600.
 660. 8.04 4.48 3.58 1449. 600.
 660. 10.79 5.965 4.229 1349. 549.6'''.lstrip()))

        receiver_mod = cake.LayeredModel.from_scanlines(
                                    cake.read_nd_model_str('''
 0. 5.8 3.46 2.6 1264. 600.
 20. 5.8 3.46 2.6 1264. 600.
 20. 6.5 3.85 2.9 1283. 600.
 35. 6.5 3.85 2.9 1283. 600.
'''.lstrip()))

        q2_store_dir = mkdtemp(prefix='gfstore')
        q_store_dir = mkdtemp(prefix='gfstore')
        self.tempdirs.append(q2_store_dir)
        self.tempdirs.append(q_store_dir)

        # qseis2d
        q2conf = qseis2d.QSeis2dConfig()
        q2conf.gf_directory = os.path.join(q2_store_dir, 'qseisS_green')

        qss = q2conf.qseis_s_config
        qss.receiver_basement_depth = 35.
        qss.calc_slowness_window = 0
        qss.slowness_window = slowness_window

        q2conf.time_region = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        q2conf.cut = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qss.sw_flat_earth_transform = 0

        config_q2 = gf.meta.ConfigTypeA(
            id='qseis2d_test_q2',
            ncomponents=10,
            sample_rate=0.1,
            receiver_depth=0. * km,
            source_depth_min=10. * km,
            source_depth_max=10. * km,
            source_depth_delta=1. * km,
            distance_min=5529. * km,
            distance_max=5531. * km,
            distance_delta=1. * km,
            modelling_code_id='qseis2d',
            earthmodel_1d=mod,
            earthmodel_receiver_1d=receiver_mod,
            tabulated_phases=[
                gf.meta.TPDef(
                    id='begin',
                    definition='p,P,p\\,P\\'),
                gf.meta.TPDef(
                    id='end',
                    definition='2.5'),
            ])

        config_q2.validate()
        gf.store.Store.create_editables(
            q2_store_dir, config=config_q2, extra={'qseis2d': q2conf})

        store = gf.store.Store(q2_store_dir, 'r')
        store.make_ttt()
        store.close()

        # build store
        try:
            qseis2d.build(q2_store_dir, nworkers=1)
        except qseis2d.QSeis2dError as e:
            if str(e).find('could not start qseis2d') != -1:
                logger.warning('qseis2d not installed; '
                               'skipping test_pyrocko_qseis_vs_qseis2d')
                logger.warning(e)
                return
            else:
                raise

        # qseis
        config_q = copy.deepcopy(config_q2)
        config_q.id = 'qseis2d_test_q'
        config_q.modelling_code_id = 'qseis.2006a'

        qconf = qseis.QSeisConfig()
        qconf.qseis_version = '2006a'

        qconf.slowness_window = slowness_window

        qconf.time_region = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qconf.cut = (
            gf.meta.Timing('0'),
            gf.meta.Timing('end+100'))

        qconf.sw_flat_earth_transform = 0

        config_q.validate()
        gf.store.Store.create_editables(
            q_store_dir, config=config_q, extra={'qseis': qconf})

        store = gf.store.Store(q_store_dir, 'r')
        store.make_ttt()
        store.close()

        # build store
        try:
            qseis.build(q_store_dir, nworkers=1)
        except qseis.QSeisError as e:
            if str(e).find('could not start qseis') != -1:
                logger.warning('qseis not installed; '
                               'skipping test_pyrocko_qseis_vs_qseis2d')
                logger.warning(e)
                return
            else:
                raise

        # forward model
        source = gf.MTSource(
            lat=0.,
            lon=0.,
            depth=10. * km)

        source.m6 = tuple(random.random() * 2. - 1. for x in range(6))

        azi = 0.    # QSeis2d only takes one receiver without azimuth variable
        dist = 5530. * km

        dnorth = dist * math.cos(azi * d2r)
        deast = dist * math.sin(azi * d2r)

        targets_q = []
        targets_q2 = []
        for cha in 'rtz':
            target_q2 = gf.Target(
                quantity='displacement',
                codes=('', '0000', 'Q2', cha),
                north_shift=dnorth,
                east_shift=deast,
                store_id='qseis2d_test_q2')

            dist = source.distance_to(target_q2)
            azi, bazi = source.azibazi_to(target_q2)

            if cha == 'r':
                target_q2.azimuth = bazi + 180.
                target_q2.dip = 0.
            elif cha == 't':
                target_q2.azimuth = bazi - 90.
                target_q2.dip = 0.
            elif cha == 'z':
                target_q2.azimuth = 0.
                target_q2.dip = 90.

            target_q = copy.deepcopy(target_q2)
            target_q.store_id = 'qseis2d_test_q'
            target_q.codes = ('', '0000', 'Q', cha)
            targets_q2.append(target_q2)
            targets_q.append(target_q)

        targets = targets_q + targets_q2
        engine = gf.LocalEngine(store_dirs=[q2_store_dir, q_store_dir])
        response = engine.process(source, targets)

        qtrcs = []
        q2trcs = []
        for s, target, trc in response.iter_results():
            if target.codes[2] == 'Q':
                qtrcs.append(trc)
            else:
                q2trcs.append(trc)

        for q, q2 in zip(qtrcs, q2trcs):
            num.testing.assert_allclose(q.ydata, q2.ydata, atol=4e-23)

#        trace.snuffle(qtrcs + q2trcs)


if __name__ == '__main__':
    util.setup_logging('test_qseis_qseis2d', 'warning')
    unittest.main()
