Plotting functions
========================================

Beachballs (focal mechanisms)
-------------------------------

Classes covered in these examples:
 * :py:class:`pyrocko.beachball` (visual representation of a focal mechanism)
 * :py:mod:`pyrocko.moment_tensor` (a 3x3 matrix representation of an
   earthquake source)
 * :py:class:`pyrocko.gf.seismosizer.DCSource` (a representation of a double
   couple source object),
 * :py:class:`pyrocko.gf.seismosizer.RectangularExplosionSource` (a
   representation of a rectangular explostion source), 
 * :py:class:`pyrocko.gf.seismosizer.CLVDSource` (a representation of a
   compensated linear vector diploe source object)
 * :py:class:`pyrocko.gf.seismosizer.DoubleDCSource` (a representation of a
   double double-couple source object).

Beachballs from moment tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to create beachballs from (random) moment tensors.  

::
    
    import random
    import logging
    import sys
    from matplotlib import pyplot as plt
    from pyrocko import beachball, moment_tensor as pmt
    from pyrocko import util

    logger = logging.getLogger(sys.argv[0])

    util.setup_logging()

    fig = plt.figure(figsize=(10., 4.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    axes = fig.add_subplot(1, 1, 1)

    for i in xrange(200):

        # create random moment tensor
        mt = pmt.MomentTensor.random_mt()

        try:
            # create beachball from moment tensor
            beachball.plot_beachball_mpl(
                mt, axes,
                # type of beachball: deviatoric, full or double couple (dc)
                beachball_type='full',
                size=random.random()*120.,
                position=(random.random()*10., random.random()*10.),
                alpha=random.random(),
                linewidth=1.0)

        except beachball.BeachballError, e:
            logger.error('%s for MT:\n%s' % (e, mt))

    axes.set_xlim(0., 10.)
    axes.set_ylim(0., 10.)
    axes.set_axis_off()
    fig.savefig('beachball-example01.pdf')

    plt.show()

.. figure :: /static/beachball-example01.png
    :align: center
    :alt: Beachballs (focal mechanisms) created by moment tensors.

    An artistic display of focal mechanisms drawn by classes
    :py:class:`pyrocko.beachball` and :py:mod:`pyrocko.moment_tensor`.


This example shows how to plot a full, a deviatoric and a double-couple beachball
for a moment tensor.

::

    from matplotlib import pyplot as plt
    from pyrocko import beachball, moment_tensor as pmt, plot
    
    fig = plt.figure(figsize=(4., 2.))
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    axes = fig.add_subplot(1, 1, 1)
    axes.set_xlim(0., 4.)
    axes.set_ylim(0., 2.)
    axes.set_axis_off()
    
    for i, beachball_type in enumerate(['full', 'deviatoric', 'dc']):
        beachball.plot_beachball_mpl(
                pmt.as_mt((124654616., 370943136., -6965434.0,
                           553316224., -307467264., 84703760.0)),
                axes,
                beachball_type=beachball_type,
                size=60.,
                position=(i+1, 1),
                color_t=plot.mpl_color('scarletred2'),
                linewidth=1.0)
    
    fig.savefig('beachball-example03.pdf')
    plt.show()

.. figure :: /static/beachball-example03.png
    :align: center
    :alt: Beachballs (focal mechanisms) options created from moment tensor

    The three types of beachballs that can be plotted through pyrocko.

Beachballs from source objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to add beachballs of various sizes to the corners of a
plot by obtaining the moment tensor from four different source object types:
:py:class:`pyrocko.gf.seismosizer.DCSource` (upper left),
:py:class:`pyrocko.gf.seismosizer.RectangularExplosionSource` (upper right), 
:py:class:`pyrocko.gf.seismosizer.CLVDSource` (lower left) and
:py:class:`pyrocko.gf.seismosizer.DoubleDCSource` (lower right).

Creating the beachball this ways allows for finer control over their location
based on their size (in display units) which allows for a round beachball even
if the axis are not 1:1.

::

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

.. figure :: /static/beachball-example02.png
    :align: center
    :alt: Beachballs (focal mechanisms) created in corners of graph.

    Four different source object types plotted with different beachball sizes.


Add station symbols to focal sphere diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shows how to add station symbols at the positions where P wave
rays pierce the focal sphere.

The function to plot focal spheres
(:py:func:`pyrocko.beachball.plot_beachball_mpl`) uses the function
:py:func:`pyrocko.beachball.project` in the final projection from 3D to 2D
coordinates. Here we use this function to place additional symbols on the plot.
The take-off angles needed can be computed with some help of the
:py:mod:`pyrocko.cake` module. Azimuth and distance computations are done with
functions from :py:mod:`pyrocko.orthodrome`.

:download:`beachball_example04.py </static/beachball_example04.py>`

::

    import numpy as num
    from matplotlib import pyplot as plt
    from pyrocko import moment_tensor as pmt, beachball, cake, orthodrome

    km = 1000.

    # source position and mechanism
    slat, slon, sdepth = 0., 0., 10.*km
    mt = pmt.MomentTensor.random_dc()

    # receiver positions
    rdepth = 0.0
    rlatlons = [(50., 10.), (60., -50.), (-30., 60.)]

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
            phases=cake.PhaseDef('P'),
            zstart=sdepth, zstop=rdepth, distances=[distance*cake.m2d])

        if not rays:
            continue

        takeoff = rays[0].takeoff_angle()
        azi = orthodrome.azimuth(slat, slon, rlat, rlon)

        # to spherical coordinates, r, theta, phi in radians
        rtp = num.array([[1., num.deg2rad(takeoff), num.deg2rad(90.-azi)]])

        # to 3D coordinates (x, y, z)
        points = beachball.numpy_rtp2xyz(rtp)

        # project to 2D with same projection as used in beachball
        x, y = beachball.project(points, projection=projection).T

        axes.plot(x, y, '+', ms=10., mew=2.0, mec='black', mfc='none')

    fig.savefig('beachball-example04.png')


.. figure :: /static/beachball-example04.png
    :align: center
    :alt: Focal sphere diagram with station symbols

    Focal sphere diagram with markers at positions of P wave ray piercing points.
