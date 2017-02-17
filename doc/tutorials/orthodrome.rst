Geodesic functions
==================

Pyrocko's :class:`~pyrocko.orthodrome` module offers geodesic functions to solve a variety of common geodetic problems, like distance and angle calculation on a spheroid or 


Distance between points on earth
----------------------------------

In this example we use :func:`~pyrocko.orthodrome.distance_accurate50m` and :class:`pyrocko.model` to calculate the distance between two points on earth.

::

    from pyrocko import orthodrome, model

    # one possibility:
    d = orthodrome.distance_accurate50m(e,s)
    print 'Distance between e and s is %g km' % (d/1000.)

    # another possibility:
    e = model.Event(lat=10., lon=20.)
    s = model.Station(lat=15., lon=120.)
    s.set_event_relative_data(e)
    print 'Distance between e and s is %g km' % (s.dist_m/1000.)


This can also be serialised for multiple coordinates stored in :class:`numpy.ndarray` through :func:`~pyrocko.orthodrome.distance_accurate50m_numpy`.

::

    import numpy as num
    from pyrocko import orthodrome

    ncoords = 1000

    # First set of coordinates
    lats_a = num.random.random_integers(-180, 180, ncoords)
    lons_a = num.random.random_integers(-90, 90, ncoords)

    # Second set of coordinates
    lats_b = num.random.random_integers(-180, 180, ncoords)
    lons_b = num.random.random_integers(-90, 90, ncoords)

    orthodrome.distance_accurate50m_numpy(lats_a, lons_a, lats_b, lons_b)


Azimuth and backazimuth
-------------------------

Calculation of azimuths and backazimuths between two points on a spherical earth.

::

    from pyrocko import orthodrome

    # For a single point
    orthodrome.azibazi(49.1, 20.5, 45.4, 22.3)

    >>> (161.05973376168285, -17.617746351508035)  # Azimuth and backazimuth

    import numpy as num
    # First set of coordinates
    lats_a = num.random.random_integers(-180, 180, ncoords)
    lons_a = num.random.random_integers(-90, 90, ncoords)

    # Second set of coordinates
    lats_b = num.random.random_integers(-180, 180, ncoords)
    lons_b = num.random.random_integers(-90, 90, ncoords)

    orthodrome.azibazi_numpy(lats_a, lons_a, lats_b, lons_b)
