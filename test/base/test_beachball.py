# python 2/3

from __future__ import division, print_function, absolute_import
import unittest
import math
import numpy as num
from io import BytesIO

from pyrocko import util, moment_tensor as mtm
from pyrocko.plot import beachball

from random import random, choice


def fuzz_angle(mi, ma):
    crits = [mi, ma, 0., 1., -1., 90., -90., 180., -180., 270., -270.]
    crits = [x for x in crits if mi <= x and x <= ma]
    ran = [
        0.,
        0.,
        0.,
        +random()*(ma-mi)*1.0e-8,
        -random()*(ma-mi)*1.0e-8,
        +random()*(ma-mi)*1.0e-4,
        -random()*(ma-mi)*1.0e-4,
        random()*(ma-mi),
        -random()*(ma-mi)]

    while True:
        v = choice(crits) + choice(ran)
        if mi <= v and v <= ma:
            return v


class BeachballTestCase(unittest.TestCase):

    def compare_beachball(self, mt, show=False, **kwargs):
        from matplotlib import pyplot as plt
        from matplotlib import image
        plt.switch_backend('Agg')

        plotargs = dict(
            size=1.0,
            size_units='data',
            linewidth=2.0,
            projection='lambert')

        plotargs.update(kwargs)

        imgs = []
        for iplot, plot in enumerate([
                beachball.plot_beachball_mpl,
                beachball.plot_beachball_mpl_pixmap]):

            fig = plt.figure(figsize=(3, 3), dpi=100)
            axes = fig.add_subplot(1, 1, 1, aspect=1.)
            axes.cla()
            axes.axison = False
            axes.set_xlim(-1.05, 1.05)
            axes.set_ylim(-1.05, 1.05)

            plot(mt, axes, **plotargs)

            f = BytesIO()
            fig.savefig(f, format='png')
            f.seek(0)
            imgs.append(image.imread(f, format='png'))
            fig.clear()
            plt.close(fig)

        a, b = imgs

        d = num.abs(a-b)
        d[:, :, 3] = 1.
        dsum = num.sum(d[:, :, :3])
        if dsum > 1600 or show:
            print(dsum)
            print(mt)
            fig = plt.figure()
            axes1 = fig.add_subplot(1, 3, 1, aspect=1.)
            axes2 = fig.add_subplot(1, 3, 2, aspect=1.)
            axes3 = fig.add_subplot(1, 3, 3, aspect=1.)
            axes1.imshow(a)
            axes2.imshow(b)
            axes3.imshow(d)
            plt.show()
            plt.close(fig)

        assert dsum <= 1600
        return dsum

    def test_random_mts(self, **kwargs):
        nx = 10
        for x in range(nx):
            m6 = num.random.random(6)*2.-1.
            m = mtm.symmat6(*m6)
            mt = mtm.MomentTensor(m=m)
            self.compare_beachball(mt, **kwargs)

        for x in range(nx):
            mt = mtm.MomentTensor.random_mt()
            self.compare_beachball(mt, **kwargs)

    def test_projections(self):
        n = 1000
        points = num.zeros((n, 3))
        phi = num.random.random(n) * math.pi * 2.0
        theta = num.random.random(n) * math.pi / 2.0

        points[:, 0] = num.cos(phi) * num.sin(theta)
        points[:, 1] = num.sin(phi) * num.sin(theta)
        points[:, 2] = num.cos(theta)

        for projection in ['lambert', 'stereographic', 'orthographic']:
            projection = 'stereographic'
            points2 = beachball.project(points, projection=projection)
            points_new = beachball.inverse_project(
                points2, projection=projection)
            assert num.all(num.abs(points - points_new) < 1e-5)

    def test_specific_mts(self):
        for m6 in [
                (1., 0., 0., 0., 0., 0.),
                (0., 1., 0., 0., 0., 0.),
                (0., 0., 1., 0., 0., 0.),
                (0., 0., 0., 1., 0., 0.),
                (0., 0., 0., 0., 1., 0.),
                (0., 0., 0., 0., 0., 1.)]:

            m = mtm.symmat6(*m6)
            mt = mtm.MomentTensor(m=m)

            self.compare_beachball(mt)

    def test_random_dcs(self):
        nx = 20

        for x in range(nx):
            strike = fuzz_angle(0., 360.)
            dip = fuzz_angle(0., 90.)
            rake = fuzz_angle(-180., 180.)

            mt = mtm.MomentTensor(
                strike=strike,
                dip=dip,
                rake=rake)

            self.compare_beachball(mt)

    def test_specific_dcs(self):
        for strike, dip, rake in [
                [270., 0.0, 0.01],
                [360., 28.373841741182012, 90.],
                [0., 0., 0.]]:

            mt = mtm.MomentTensor(
                strike=strike,
                dip=dip,
                rake=rake)

            self.compare_beachball(mt)

    def test_random_mts_views(self):
        for view in ('top', 'north', 'south', 'east', 'west'):
            self.test_random_mts(view=view)

    @unittest.skip('contour and contourf do not support transform')
    def test_plotstyle(self):

        # contour and contourf do not support transform
        mt = mtm.MomentTensor.random_mt()
        self.compare_beachball(
            mt, show=True, size=20., size_units='points', position=(0.5, 0.5))

    def test_fuzzy_beachball(self):

        from matplotlib import pyplot as plt

        def get_random_uniform(lower, upper):
            return lower + (upper - lower) * random()

        fig = plt.figure(figsize=(4., 4.))
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        axes = fig.add_subplot(1, 1, 1)

        strike = 135.
        dip = 65.
        rake = 15.

        best_mt = mtm.MomentTensor.from_values((strike, dip, rake))

        n_balls = 1000
        mts = []
        for i in range(n_balls):
            strike_dev = get_random_uniform(-15., 15.)
            mts.append(mtm.MomentTensor.from_values(
                (strike + strike_dev, dip, rake)))

        kwargs = {
            'beachball_type': 'full',
            'size': 8,
            'position': (5, 5),
            'color_t': 'black',
            'edgecolor': 'black'}

        beachball.plot_fuzzy_beachball_mpl_pixmap(mts, axes, best_mt, **kwargs)

        axes.set_xlim(0., 10.)
        axes.set_ylim(0., 10.)
        axes.set_axis_off()

        # fig.savefig('fuzzy_bb_no.png', dpi=400)
        # plt.show()

    def show_comparison(self, nx=10):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        if nx > 1:
            plt.ion()
            plt.show()

        fig = plt.figure()
        axes1 = fig.add_subplot(2, 3, 1, aspect=1.)
        axes2 = fig.add_subplot(2, 3, 2, aspect=1.)
        axes3 = fig.add_subplot(2, 3, 3, aspect=1.)
        axes4 = fig.add_subplot(2, 3, 4, projection='3d', aspect=1.)
        axes5 = fig.add_subplot(2, 3, 5, projection='3d', aspect=1.)

        try:
            import mopad
        except ImportError:
            mopad = None

        for x in range(nx):
            mt = mtm.MomentTensor.random_mt()
            mt = mt.deviatoric()

            for axes in (axes1, axes2, axes3):
                axes.cla()
                axes.axison = False
                axes.set_xlim(-1.05, 1.05)
                axes.set_ylim(-1.05, 1.05)

            for axes in (axes4, axes5):
                axes.cla()

            axes1.set_title('Copacabana')
            axes2.set_title('Contour')
            axes3.set_title('MoPaD')
            axes4.set_title('Patches')
            axes5.set_title('Lines')

            beachball.plot_beachball_mpl(mt, axes1, size_units='data')
            beachball.plot_beachball_mpl_pixmap(mt, axes2, size_units='data')

            beachball.plot_beachball_mpl_construction(
                mt, axes4, show='patches')
            beachball.plot_beachball_mpl_construction(
                mt, axes5, show='lines')

            if mopad:
                try:
                    mop_mt = mopad.MomentTensor(M=mt.m6())
                    mop_beach = mopad.BeachBall(mop_mt)
                    kwargs = dict(
                        plot_projection='lambert',
                        plot_nodalline_width=2,
                        plot_faultplane_width=2,
                        plot_outerline_width=2)

                    mop_beach.ploBB(kwargs, ax=axes3)

                except Exception:
                    print(
                        'mopad failed (maybe patched mopad version is needed')

            fig.canvas.draw()

        if nx == 1:
            plt.show()


if __name__ == "__main__":
    util.setup_logging('test_beachball', 'warning')
    unittest.main()
