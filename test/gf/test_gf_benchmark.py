from __future__ import division, print_function, absolute_import
from builtins import range, zip

import unittest
import numpy as num
import cProfile
import math
import logging
import shutil

from tempfile import mkdtemp
from ..common import Benchmark
from pyrocko import gf, util

random = num.random
logger = logging.getLogger('pyrocko.test.test_gf_benchmark')
benchmark = Benchmark()

r2d = 180. / math.pi
d2r = 1.0 / r2d
km = 1000.


class GFBenchmarkTest(unittest.TestCase):

    tempdirs = []

    def __init__(self, *args, **kwargs):
        self._dummy_store = None
        unittest.TestCase.__init__(self, *args, **kwargs)

    @classmethod
    def tearDownClass(cls):
        for d in cls.tempdirs:
            shutil.rmtree(d)

    def dummy_store(self):
        if self._dummy_store is None:

            conf = gf.ConfigTypeA(
                id='empty_regional',
                source_depth_min=0.,
                source_depth_max=20*km,
                source_depth_delta=1*km,
                distance_min=1*km,
                distance_max=2000*km,
                distance_delta=1*km,
                sample_rate=2.0,
                ncomponents=10)

            store_dir = mkdtemp(prefix='gfstore')
            self.tempdirs.append(store_dir)

            gf.Store.create(store_dir, config=conf)
            self._dummy_store = gf.Store(store_dir)

        return self._dummy_store

    def setUp(self):
        self.cprofile = cProfile.Profile()
        self.cprofile.enable()
        self.addCleanup(
            lambda: self.cprofile.dump_stats('/tmp/make_sum_params.cprof'))

    def _test_sum_benchmark(self):
        from pyrocko.gf import store_ext
        benchmark.show_factor = True

        def test_weights_bench(store, dim, ntargets, interpolation):
            source = gf.RectangularSource(
                lat=1.4, lon=-1.6,
                depth=10*km, north_shift=0.1, east_shift=0.1,
                width=dim, length=dim)

            targets = [gf.Target(
                lat=random.random()*10.,
                lon=random.random()*10.,
                north_shift=0.1,
                east_shift=0.1) for x in range(ntargets)]

            dsource = source.discretize_basesource(store, targets[0])
            source_coords_arr = dsource.coords5()
            mts_arr = dsource.m6s

            receiver_coords_arr = num.empty((len(targets), 5))
            for itarget, target in enumerate(targets):
                receiver = target.receiver(store)
                receiver_coords_arr[itarget, :] = \
                    [receiver.lat, receiver.lon, receiver.north_shift,
                     receiver.east_shift, receiver.depth]
            ns = mts_arr.shape[0]
            label = '_ns%04d_nt%04d_%s' % (ns,
                                           len(targets),
                                           interpolation)

            @benchmark.labeled('c%s' % label)
            def sum_c():
                return store_ext.make_sum_params(
                    store.cstore,
                    source_coords_arr,
                    mts_arr,
                    receiver_coords_arr,
                    'elastic10',
                    interpolation,
                    0)

            @benchmark.labeled('p%s' % label)
            def sum_python():
                weights_c = []
                irecords_c = []
                for itar, target in enumerate(targets):
                    receiver = target.receiver(store)
                    dsource = source.discretize_basesource(store, target)

                    for i, (component, args, delays, weights) in \
                            enumerate(store.config.make_sum_params(
                                dsource, receiver)):
                        if len(weights_c) <= i:
                            weights_c.append([])
                            irecords_c.append([])

                        if interpolation == 'nearest_neighbor':
                            irecords = num.array(store.config.irecords(*args))
                            weights = num.array(weights)
                        else:
                            assert interpolation == 'multilinear'
                            irecords, ip_weights =\
                                store.config.vicinities(*args)
                            neach = irecords.size // args[0].size
                            weights = num.repeat(weights, neach) * ip_weights
                            delays = num.repeat(delays, neach)

                        weights_c[i].append(weights)
                        irecords_c[i].append(irecords)
                for c in range(len(weights_c)):
                    weights_c[c] = num.concatenate([w for w in weights_c[c]])
                    irecords_c[c] = num.concatenate([ir for
                                                     ir in irecords_c[c]])

                return list(zip(weights_c, irecords_c))

            rc = sum_c()
            rp = sum_python()

            print(benchmark.__str__(header=False))
            benchmark.clear()

            # Comparing the results
            if isinstance(store.config, gf.meta.ConfigTypeA):
                idim = 4
            elif isinstance(store.config, gf.meta.ConfigTypeB):
                idim = 8
            if interpolation == 'nearest_neighbor':
                idim = 1

            for i, nsummands in enumerate([6, 6, 4]):
                for r in [0, 1]:
                    r_c = rc[i][r]
                    r_p = rp[i][r].reshape(ntargets, nsummands, ns*idim)
                    r_p = num.transpose(r_p, axes=[0, 2, 1])

                    num.testing.assert_almost_equal(r_c, r_p.flatten())
                if False:
                    print('irecord_c: {0:>7}, {1:>7}'.format(
                        rc[i][1].min(), rc[i][1].max()))
                    print('irecord_p: {0:>7}, {1:>7}'.format(
                        rp[i][1].min(), rp[i][1].max()))

                if False:
                    print('weights_c: {0:>7}, {1:>7}'.format(
                        rc[i][0].min(), rc[i][0].max()))
                    print('weights_p: {0:>7}, {1:>7}'.format(
                        rp[i][0].min(), rp[i][0].max()))

        '''
        Testing loop
        '''
        dims = [2*km, 5*km, 8*km, 16*km]
        ntargets = [10, 100, 1000]

        dims = [16*km]
        ntargets = [1000]
        store = self.dummy_store()
        store.open()

        for interpolation in ['multilinear', 'nearest_neighbor']:
            for d in dims:
                for nt in ntargets:
                    test_weights_bench(store, d, nt, interpolation)


if __name__ == '__main__':
    util.setup_logging('test_gf', 'warning')
    unittest.main(defaultTest='GFBenchmarkTest.test_sum_benchmark')
