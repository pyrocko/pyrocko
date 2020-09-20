from __future__ import division, print_function, absolute_import
import unittest
import shutil
import tempfile
from pyrocko import util, gf
from pyrocko.fomosto import qssp


@unittest.skipUnless(
    qssp.have_backend(), 'backend qssp not available')
class QSSPTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='pyrocko.qssp')

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @unittest.skipUnless(
        'qssp.2010' in qssp.have_backend(), 'backend qssp.2010 not available')
    def test_qssp_build_2010(self):
        qssp.init(self.tmpdir, '2010', config_params=dict(
            source_depth_max=10e3,
            distance_min=500e3,
            distance_max=600e3))
        store = gf.store.Store(self.tmpdir, 'r')
        store.make_ttt()
        qssp.build(self.tmpdir)

        engine = gf.LocalEngine(store_dirs=[self.tmpdir])

        source = gf.DCSource(
            lat=0.,
            lon=0.,
            depth=10e3,
            magnitude=6.0)

        targets = [gf.Target(
            quantity='displacement',
            codes=('', 'STA', '', comp),
            lat=0.,
            lon=0.,
            north_shift=500e3,
            east_shift=100e3) for comp in 'NEZ']

        engine.process(source, targets)

    @unittest.skipUnless(
        'qssp.2017' in qssp.have_backend(), 'backend qssp.2017 not available')
    def test_qssp_build_2017_rotational(self):
        qssp.init(self.tmpdir, '2017', config_params=dict(
            stored_quantity='rotation',
            source_depth_max=10e3,
            distance_min=500e3,
            distance_max=600e3))

        store = gf.store.Store(self.tmpdir, 'r')
        store.make_ttt()
        qssp.build(self.tmpdir)

        del store

        engine = gf.LocalEngine(store_dirs=[self.tmpdir])

        source = gf.DCSource(
            lat=0.,
            lon=0.,
            depth=10e3,
            magnitude=6.0)

        targets = [gf.Target(
            quantity='rotation',
            codes=('', 'ROT', '', comp),
            lat=0.,
            lon=0.,
            north_shift=500e3,
            east_shift=100e3) for comp in 'NEZ']

        engine.process(source, targets)

    @unittest.skipUnless(
        'qssp.2020' in qssp.have_backend(), 'backend qssp.2020 not available')
    def test_qssp_build_2020_rotational(self):
        qssp.init(self.tmpdir, '2020', config_params=dict(
            stored_quantity='rotation',
            source_depth_max=10e3,
            distance_min=500e3,
            distance_max=600e3))

        store = gf.store.Store(self.tmpdir, 'r')
        store.make_ttt()
        qssp.build(self.tmpdir)

        del store

        engine = gf.LocalEngine(store_dirs=[self.tmpdir])

        source = gf.DCSource(
            lat=0.,
            lon=0.,
            depth=10e3,
            magnitude=6.0)

        targets = [gf.Target(
            quantity='rotation',
            codes=('', 'ROT', '', comp),
            lat=0.,
            lon=0.,
            north_shift=500e3,
            east_shift=100e3) for comp in 'NEZ']

        engine.process(source, targets)


if __name__ == '__main__':
    util.setup_logging('test_gf_qssp', 'warning')
    unittest.main()
