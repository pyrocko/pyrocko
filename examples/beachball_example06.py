#!/usr/bin/env python3

from matplotlib import pyplot as plt
from pyrocko import moment_tensor as pmt
from pyrocko.plot import beachball

mt = pmt.as_mt([0.424, -0.47, 0.33, 0.711, -0.09, 0.16])
axes = plt.gca()

beachball.plot_beachball_mpl(
    mt, axes,
    size=50.,
    position=(0., 0.),
    view='top')

beachball.plot_beachball_mpl(
    mt, axes,
    size=50.,
    position=(0, -1.),
    view='south')

beachball.plot_beachball_mpl(
    mt, axes,
    size=50.,
    position=(-1, 0.),
    view='east')

beachball.plot_beachball_mpl(
    mt, axes,
    size=50.,
    position=(0, 1.),
    view='north')

beachball.plot_beachball_mpl(
    mt, axes,
    size=50.,
    position=(1, 0.),
    view='west')

axes.set_xlim(-2., 2.)
axes.set_ylim(-2., 2.)
plt.show()
