Geodesic functions
==================

Pyrocko's :class:`~pyrocko.orthodrome` module offers geodesic functions to solve a variety of common geodetic problems, like distance and angle calculation on a spheroid.


Distance between points on earth
----------------------------------

In this example we use :func:`~pyrocko.orthodrome.distance_accurate50m` and :class:`pyrocko.model` to calculate the distance between two points on earth.

::

    from pyrocko import orthodrome, model

    e = model.Event(lat=10., lon=20.)
    s = model.Station(lat=15., lon=120.)

    # one possibility:
    d = orthodrome.distance_accurate50m(e, s)
    print 'Distance between e and s is %g km' % (d/1000.)

    # another possibility:
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

    ncoords = 1000
    # First set of coordinates
    lats_a = num.random.random_integers(-180, 180, ncoords)
    lons_a = num.random.random_integers(-90, 90, ncoords)

    # Second set of coordinates
    lats_b = num.random.random_integers(-180, 180, ncoords)
    lons_b = num.random.random_integers(-90, 90, ncoords)

    orthodrome.azibazi_numpy(lats_a, lons_a, lats_b, lons_b)


Conversion to carthesian coordinates
------------------------------------

Given two sets of coordinates, :func:`~pyrocko.orthodrome.latlon_to_ne` returns the distance in meters in north/east direction.

::

    from pyrocko import orthodrome

    # option 1, coordinates as floats
    north_m, east_m = orthodrome.latlon_to_ne(
        10.3,   # origin latitude
        12.4,   # origin longitude
        10.5,   # target latitude
        12.6)   # target longitude

    print north_m, east_m

    >>> 22199.7843582 21821.3511789

    # option 2, coordinates from instances with 'lon' and 'lat' attributes

    from pyrocko.gf import seismosizer

    source = seismosizer.DCSource(lat=10.3, lon=12.4)
    target = seismosizer.Target(lat=10.5, lon=12.6)

    north_m, east_m = orthodrome.latlon_to_ne(source, target)

    print north_m, east_m

    >>> 22199.7843582 21821.3511789


Relative carthesian coordinates to latitude and longitude
---------------------------------------------------------

::

    from pyrocko import orthodrome

    # arguments: origin lat, origin lon, north [m], east [m]
    print "latitude: %s, longitude: %s " % orthodrome.ne_to_latlon(10.3, 12.4, 22200., 21821.)

    >>> latitude: 10.4995878932, longitude: 12.5995823469
