import numpy as num
from matplotlib import pyplot as plt
from pyrocko import moment_tensor as pmt, cake, orthodrome
from pyrocko.plot import beachball

km = 1000.

# source position and mechanism
slat, slon, sdepth = 0., 0., 10.*km
mt = pmt.MomentTensor.random_dc()

# receiver positions
rdepth = 0.0
rlatlons = [
    (50., 10.),
    (60., -50.),
    (-30., 60.),
    (0.8, 0.8),  # first arrival is P  -> takeoff angle < 90deg
    (0.5, 0.5),  # first arrival is p  -> takeoff angle > 90deg
]

# earth model and phase for takeoff angle computations
mod = cake.load_model('ak135-f-continental.m')
phases = cake.PhaseDef.classic('P')

# setup figure with aspect=1.0/1.0, ranges=[-1.1, 1.1]
fig = plt.figure(figsize=(2., 2.))  # size in inch
fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
axes = fig.add_subplot(1, 1, 1, aspect=1.0)
axes.set_axis_off()
axes.set_xlim(-1.1, 1.1)
axes.set_ylim(-1.1, 1.1)

projection = 'lambert'

beachball.plot_beachball_mpl(
    mt, axes,
    position=(0., 0.),
    size=2.0,
    color_t=(0.7, 0.4, 0.4),
    projection=projection,
    size_units='data')

for rlat, rlon in rlatlons:
    distance = orthodrome.distance_accurate50m(slat, slon, rlat, rlon)
    rays = mod.arrivals(
        phases=[cake.PhaseDef('P'), cake.PhaseDef('p')],
        zstart=sdepth, zstop=rdepth, distances=[distance*cake.m2d])

    if not rays:
        continue

    takeoff = rays[0].takeoff_angle()
    azi = orthodrome.azimuth(slat, slon, rlat, rlon)
    polarity = num.sign(beachball.amplitudes(mt, [azi], [takeoff]))

    # to spherical coordinates, r, theta, phi in radians
    # flip direction when takeoff is upward
    rtp = num.array([[
        1.0 if takeoff <= 90. else -1.,
        num.deg2rad(takeoff),
        num.deg2rad(90.-azi)]])
    # to 3D coordinates (x, y, z)
    points = beachball.numpy_rtp2xyz(rtp)

    # project to 2D with same projection as used in beachball
    x, y = beachball.project(points, projection=projection).T

    axes.plot(
        x, y,
        '+' if polarity > 0.0 else 'x',
        ms=10. if polarity > 0.0 else 10./num.sqrt(2.),
        mew=2.0,
        mec='black',
        mfc='none')

fig.savefig('beachball-example04.png')
