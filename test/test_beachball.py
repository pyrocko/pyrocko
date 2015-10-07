import unittest
import numpy as num
from cStringIO import StringIO as StringIO

from pyrocko import util, moment_tensor as mtm, beachball

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

    def compare_beachball(self, mt):
        from matplotlib import pyplot as plt
        from matplotlib import image

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

            plot(mt, axes)

            f = StringIO()
            fig.savefig(f, format='png')
            f.seek(0)
            imgs.append(image.imread(f, format='png'))
            fig.clear()
            plt.close(fig)

        a, b = imgs

        d = num.abs(a-b)
        d[:, :, 3] = 1.
        dsum = num.sum(d[:, :, :3])
        if dsum > 1600:
            print dsum
            print mt
            fig = plt.figure()
            axes1 = fig.add_subplot(1, 3, 1, aspect=1.)
            axes2 = fig.add_subplot(1, 3, 2, aspect=1.)
            axes3 = fig.add_subplot(1, 3, 3, aspect=1.)
            axes1.imshow(a)
            axes2.imshow(b)
            axes3.imshow(d)
            plt.show()
            plt.close(fig)

    def test_random_mts(self):
        nx = 2
        for x in range(nx):
            m6 = num.random.random(6)*2.-1.
            m = mtm.symmat6(*m6)
            mt = mtm.MomentTensor(m=m)
            self.compare_beachball(mt)

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
        nx = 2

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


if __name__ == "__main__":
    util.setup_logging('test_moment_tensor', 'warning')
    unittest.main()
