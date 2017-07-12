from matplotlib import transforms, pyplot as plt
from pyrocko import beachball, gf

# create source object
source1 = gf.DCSource(depth=35e3, strike=0., dip=90., rake=0.)

# set size of beachball
sz = 20.
# set beachball offset in points (one point from each axis)
szpt = (sz / 2.) / 72. + 1. / 72.

fig = plt.figure(figsize=(10., 4.))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# get the bounding point (left-top)
x0 = ax.get_xlim()[0]
y1 = ax.get_ylim()[1]

# create a translation matrix, based on the final figure size and
# beachball location
move_trans = transforms.ScaledTranslation(szpt, -szpt, fig.dpi_scale_trans)

# get the inverse matrix for the axis where the beachball will be plotted
inv_trans = ax.transData.inverted()

# set the bouding point relative to the plotted axis of the beachball
x0, y1 = inv_trans.transform(move_trans.transform(
    ax.transData.transform((x0, y1))))

# plot beachball
beachball.plot_beachball_mpl(source1.pyrocko_moment_tensor(), ax,
                             beachball_type='full', size=sz,
                             position=(x0, y1), linewidth=1.)


# create source object
source2 = gf.RectangularExplosionSource(depth=35e3, strike=0., dip=90.)

# set size of beachball
sz = 30.
# set beachball offset in points (one point from each axis)
szpt = (sz / 2.) / 72. + 1. / 72.

# get the bounding point (right-upper)
x1 = ax.get_xlim()[1]
y1 = ax.get_ylim()[1]

# create a translation matrix, based on the final figure size and
# beachball location
move_trans = transforms.ScaledTranslation(-szpt, -szpt, fig.dpi_scale_trans)

# get the inverse matrix for the axis where the beachball will be plotted
inv_trans = ax.transData.inverted()

# set the bouding point relative to the plotted axis of the beachball
x1, y1 = inv_trans.transform(move_trans.transform(
    ax.transData.transform((x1, y1))))

# plot beachball
beachball.plot_beachball_mpl(source2.pyrocko_moment_tensor(), ax,
                             beachball_type='full', size=sz,
                             position=(x1, y1), linewidth=1.)


# create source object
source3 = gf.CLVDSource(amplitude=35e6, azimuth=30., dip=30.)

# set size of beachball
sz = 40.
# set beachball offset in points (one point from each axis)
szpt = (sz / 2.) / 72. + 1. / 72.

# get the bounding point (left-bottom)
x0 = ax.get_xlim()[0]
y0 = ax.get_ylim()[0]

# create a translation matrix, based on the final figure size and
# beachball location
move_trans = transforms.ScaledTranslation(szpt, szpt, fig.dpi_scale_trans)

# get the inverse matrix for the axis where the beachball will be plotted
inv_trans = ax.transData.inverted()

# set the bouding point relative to the plotted axis of the beachball
x0, y0 = inv_trans.transform(move_trans.transform(
    ax.transData.transform((x0, y0))))

# plot beachball
beachball.plot_beachball_mpl(source3.pyrocko_moment_tensor(), ax,
                             beachball_type='full', size=sz,
                             position=(x0, y0), linewidth=1.)

# create source object
source4 = gf.DoubleDCSource(depth=35e3, strike1=0., dip1=90., rake1=0.,
                            strike2=45., dip2=74., rake2=0.)

# set size of beachball
sz = 50.
# set beachball offset in points (one point from each axis)
szpt = (sz / 2.) / 72. + 1. / 72.

# get the bounding point (right-bottom)
x1 = ax.get_xlim()[1]
y0 = ax.get_ylim()[0]

# create a translation matrix, based on the final figure size and
# beachball location
move_trans = transforms.ScaledTranslation(-szpt, szpt, fig.dpi_scale_trans)

# get the inverse matrix for the axis where the beachball will be plotted
inv_trans = ax.transData.inverted()

# set the bouding point relative to the plotted axis of the beachball
x1, y0 = inv_trans.transform(move_trans.transform(
    ax.transData.transform((x1, y0))))

# plot beachball
beachball.plot_beachball_mpl(source4.pyrocko_moment_tensor(), ax,
                             beachball_type='full', size=sz,
                             position=(x1, y0), linewidth=1.)

fig.savefig('beachball-example02.pdf')
plt.show()
