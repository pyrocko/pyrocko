from __future__ import division, print_function, absolute_import

import sys
import os
import logging
import unittest
import numpy as num

from pyrocko import util
from pyrocko import eikonal_ext

from . import common

assert_ae = num.testing.assert_almost_equal


logger = logging.getLogger('pyrocko.test.test_eikonal')

benchmark = common.Benchmark()

km = 1000.


show_plot = int(os.environ.get('MPL_SHOW', 0))


def is_64bits():
    return sys.maxsize > 2**32


class EikonalTestCase(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        print(benchmark)

    def fpath_reference(self, fn):
        try:
            return common.test_data_file(fn)
        except util.DownloadError:
            return common.test_data_file_no_download(fn)

    def compare_with_reference(self, arr, fn, rtol=1e-9):
        fpath = self.fpath_reference(fn)
        if not os.path.exists(fpath):
            num.save(fpath, arr)

        arr_ref = num.load(fpath)
        finite_ref = num.isfinite(arr_ref)
        finite = num.isfinite(arr)

        assert num.allclose(arr_ref[finite_ref], arr[finite], rtol=rtol)

    def test_empty(self):

        speeds = num.zeros((0, 0))
        times = num.zeros((0, 0))
        delta = 1.0

        with self.assertRaises(eikonal_ext.EikonalExtError):
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

    def test_nd(self):
        ndim_max = 3
        n = 51
        for ndim in range(1, ndim_max+2):
            shape = (n,) * ndim
            speeds = num.ones(shape)
            times = num.zeros(shape) - 1.0
            delta = 2.0 / (n-1)

            iseed = tuple(i//2 for i in shape)
            times[iseed] = 0.0

            if ndim <= ndim_max:
                @benchmark.labeled('test_nd: %iD' % ndim)
                def run():
                    eikonal_ext.eikonal_solver_fmm_cartesian(
                        speeds, times, delta)

                run()

                fshapes = []
                for idim in range(ndim):
                    fshape = [1] * ndim
                    fshape[idim] = n
                    fshapes.append(tuple(fshape))

                xs = [
                    num.linspace(-1.0, 1.0, n).reshape(fshape)
                    for fshape in fshapes]

                times_ref = num.sqrt(sum(x**2 for x in xs))
                assert num.max(num.abs(times-times_ref)) \
                    < 1e6 + (ndim-1) * delta

            else:
                with self.assertRaises(eikonal_ext.EikonalExtError):
                    eikonal_ext.eikonal_solver_fmm_cartesian(
                        speeds, times, delta)

    def test_earthmodel(self):
        from pyrocko import cake
        mod = cake.load_model()

        nx, ny = 500, 500
        delta = (cake.earthradius * 2.0) / (nx-1)

        x = num.arange(nx) * delta - cake.earthradius
        y = num.arange(ny) * delta - cake.earthradius

        x2 = x[num.newaxis, :]
        y2 = y[:, num.newaxis]

        z = cake.earthradius - num.sqrt(x2**2 + y2**2)

        vp_pro = mod.profile('vp')
        z_pro = mod.profile('z')

        vp = num.interp(z, z_pro, vp_pro)

        inside = z > 0.0

        speeds = num.ones((ny, nx))
        speeds[:, :] = 300.
        speeds[inside] = vp[inside]

        times = num.zeros((ny, nx)) - 1.0

        iy = ny - int(round(600*km / delta))
        ix = nx//2

        times[iy, ix] = 0.0

        @benchmark.labeled('test_earthmodel')
        def run():
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

        run()

        times[num.logical_not(inside)] = num.nan
        self.compare_with_reference(times, 'test_earthmodel.npy')

        if show_plot:
            from matplotlib import pyplot as plt

            plt.gcf().add_subplot(1, 1, 1, aspect=1.0)
            plt.pcolormesh(x, y, speeds, cmap='gray', edgecolor='none')
            plt.contour(x, y, times, levels=num.linspace(0., 1200, 20))
            plt.gca().axis('off')
            plt.show()

    def test_2d(self):

        nx, ny = 1000, 500
        delta = 20. / float(nx)
        x = num.arange(nx) * delta - 10.0
        y = num.arange(ny) * delta - 5.0

        x2 = x[num.newaxis, :]
        y2 = y[:, num.newaxis]
        speeds = num.ones((ny, nx))  # + num.sin(x2)*0.1
        r1 = num.sqrt((x2-3)**2 + y2**2)
        r2 = num.sqrt((x2+3)**2 + y2**2)

        speeds[r1 < 2.0] = 1.5
        speeds[r2 < 2.0] = 0.66

        rstate = num.random.RandomState(23)
        speeds *= rstate.uniform(0.5, 1.5, (ny, nx))
        times = num.zeros((ny, nx)) - 1.0
        # times[:, 0] = y**2 / 10.0
        # times[:, 0] = y*10
        times[0, :] = (x-num.min(x)) * 0.2
        # times[0, 0] = 0.0
        # times[0, -1] = 0.0
        # times[ny//2, nx//2] = 0.0

        @benchmark.labeled('test_2d')
        def run():
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

        run()

        self.compare_with_reference(
            times, 'test_2d.npy',
            rtol=(1e-9 if is_64bits() else 1e-2))  # should be investigated

        if show_plot:
            from matplotlib import pyplot as plt

            plt.gcf().add_subplot(1, 1, 1, aspect=1.0)
            plt.contourf(x, y, times)
            plt.contourf(x, y, speeds, alpha=0.1, cmap='gray', )
            plt.show()

    def test_2d_snail(self):

        nx, ny = 1000, 500
        delta = 20. / float(nx)
        x = num.arange(nx) * delta - 10.0
        y = num.arange(ny) * delta - 5.0

        # x2 = x[num.newaxis, :]
        # y2 = y[:, num.newaxis]
        speeds = num.ones((ny, nx))  # + num.sin(x2)*0.1

        a = num.linspace(0.5, 4.0, 6000)

        sx = a * num.sin(a*(2.*num.pi))
        sy = a * num.cos(a*(2.*num.pi))

        isx = nx//2 + num.round(sx / delta).astype(num.int)
        isy = ny//2 + num.round(sy / delta).astype(num.int)

        speeds[isy, isx] = 0.00001

        times = num.zeros((ny, nx)) - 1.0
        times[ny//2, 0] = 0.0

        @benchmark.labeled('test_2d_snail')
        def run():
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

        run()

        self.compare_with_reference(times, 'test_2d_snail.npy')

        if show_plot:
            from matplotlib import pyplot as plt

            plt.gcf().add_subplot(1, 1, 1, aspect=1.0)
            times[times > 50.] = num.nan
            plt.contourf(
                x, y, times,
                levels=num.linspace(0., 50., 50))
            plt.contour(
                x, y, times,
                levels=num.linspace(0., 50., 50),
                colors='white',
                alpha=0.5)
            plt.contourf(
                x, y, speeds,
                alpha=0.1, cmap='gray')

            plt.show()

    def test_3d(self):

        nx, ny, nz = 100, 100, 100
        delta = 10. / float(nx)
        x = num.arange(nx) * delta - 5.0
        y = num.arange(ny) * delta - 5.0
        z = num.arange(nz) * delta - 5.0

        # x2 = x[num.newaxis, num.newaxis, :]
        # y2 = y[num.newaxis, :, num.newaxis]
        # z2 = z[:, num.newaxis, num.newaxis]

        speeds = num.ones((nz, ny, nx))  # + num.sin(x2)*0.1

        times = num.zeros((ny, nx, nz)) - 1.0
        times[nz//2, ny//2, nx//2] = 0.0

        @benchmark.labeled('test_3d')
        def run():
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times, delta)

        run()

        self.compare_with_reference(times, 'test_3d.npy')

        if show_plot:
            from mpl_toolkits.mplot3d import axes3d  # noqa
            from matplotlib import pyplot as plt

            x3, y3 = num.meshgrid(x, y)

            ax = plt.gca(projection='3d')
            ax.scatter(
                x3.flatten(), y3.flatten(), z[0], c=times[0, :, :].flatten())

            x3, z3 = num.meshgrid(x, z)

            ax = plt.gca(projection='3d')
            ax.scatter(
                x3.flatten(), y[0], z3.flatten(), c=times[:, 0, :].flatten())
            y3, z3 = num.meshgrid(y, z)

            ax = plt.gca(projection='3d')
            ax.scatter(
                x[0], y3.flatten(), z3.flatten(), c=times[:, :, 0].flatten())
            plt.show()

    def benchmark_against_fast_sweeping(self):
        from beat.fast_sweeping import fast_sweep

        nx, ny = 300, 300
        delta = 20. / float(nx)
        x = num.arange(nx) * delta
        y = num.arange(ny) * delta

        # x2 = x[num.newaxis, :]
        # y2 = y[:, num.newaxis]
        speeds = num.ones((ny, nx))  # + num.sin(x2)*0.1
        speeds[ny//2:, :] = 3.0

        slownesses = 1.0/speeds

        times_fm = num.zeros((ny, nx)) - 1.0
        times_fm[ny//4, nx//4] = 0.0

        @benchmark.labeled('fast marching')
        def fm():
            eikonal_ext.eikonal_solver_fmm_cartesian(speeds, times_fm, delta)

        times_fs = []

        @benchmark.labeled('fast sweeping')
        def fs():
            times_fs.append(fast_sweep.get_rupture_times_c(
                slownesses.flatten(), delta,
                nx, ny, ny//4, nx//4).reshape((ny, nx)))

        fm()
        fs()

        times_fs = times_fs[0]

        from matplotlib import pyplot as plt

        def get_elapsed(name):
            for xname, elapsed in benchmark.results:
                if name == xname:
                    return elapsed

        plt.gcf().add_subplot(1, 3, 1, aspect=1.0)
        plt.title('fast marching, max=%g, t_cpu=%.2g s' % (
            num.max(times_fm), get_elapsed('fast marching')))
        plt.contourf(x, y, times_fm)

        plt.gcf().add_subplot(1, 3, 2, aspect=1.0)
        plt.title('fast sweeping, max=%g, t_cpu=%.2g s' % (
            num.max(times_fs), get_elapsed('fast sweeping')))
        plt.contourf(x, y, times_fs)

        plt.gcf().add_subplot(1, 3, 3, aspect=1.0)
        plt.title('difference, |max|=%g' % num.max(num.abs(times_fm-times_fs)))
        plt.contourf(x, y, times_fm-times_fs)

        plt.show()


if __name__ == '__main__':
    util.setup_logging('test_eikonal', 'warning')
    unittest.main()
