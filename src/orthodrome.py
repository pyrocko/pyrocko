import math
import numpy as num
from pyrocko.config import config

d2r = math.pi/180.
r2d = 1./d2r
earth_oblateness = 1./298.257223563
earthradius_equator = 6378.14 * 1000.
earthradius = config().earthradius
d2m = earthradius_equator*math.pi/180.
m2d = 1./d2m


class Loc:
    '''Simple location representation

        :attrib lat: Latitude degree
        :attrib lon: Longitude degree
    '''
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon


def clip(x, mi, ma):
    ''' Clipping data array ``x``

    :param x: Continunous data to be clipped
    :param mi: Clip minimum
    :param ma: Clip maximum
    :type x: :py:class:`numpy.ndarray`
    :type mi: float
    :type ma: float

    :return: Clipped data
    :rtype: :py:class:`numpy.ndarray`
    '''
    return num.minimum(num.maximum(mi, x), ma)


def wrap(x, mi, ma):
    '''Wrapping continuous data to fundamental phase values.

    .. math::
        x_{\\mathrm{wrapped}} = x_{\\mathrm{cont},i} - \
            \\frac{ x_{\\mathrm{cont},i} - r_{\\mathrm{min}} }\
                  { r_{\\mathrm{max}} -  r_{\\mathrm{min}}} \
            \cdot ( r_{\\mathrm{max}} -  r_{\\mathrm{min}}),\ \quad
        x_{\\mathrm{wrapped}}\; \in
            \;[ r_{\\mathrm{min}},\, r_{\\mathrm{max}}].

    :param x: Continunous data to be wrapped
    :param mi: Minimum value of wrapped data
    :param ma: Maximum value of wrapped data
    :type x: :py:class:`numpy.ndarray`
    :type mi: float
    :type ma: float

    :return: Wrapped data
    :rtype: :py:class:`numpy.ndarray`
    '''
    return x - num.floor((x-mi)/(ma-mi)) * (ma-mi)


def _latlon_pair(args):
    if len(args) == 2:
        a, b = args
        return a.lat, a.lon, b.lat, b.lon

    elif len(args) == 4:
        return args


def cosdelta(*args):
    '''Cosine of the angular distance between two points ``a`` and ``b`` on
    a sphere.

    This function (find implementation below) returns the cosine of the
    distance angle 'delta' between two points ``a`` and ``b``, coordinates of
    which are expected to be given in geographical coordinates and in degrees.
    For numerical stability a maximum of 1.0 is enforced.

    .. math::

        A_{\\mathrm{lat'}} = \\frac{ \pi}{180} \cdot A_{lat}, \quad \
        A_{\\mathrm{lon'}} = \\frac{ \pi}{180} \cdot A_{lon}, \quad \
        B_{\\mathrm{lat'}} = \\frac{ \pi}{180} \cdot B_{lat}, \quad \
        B_{\\mathrm{lon'}} = \\frac{ \pi}{180} \cdot B_{lon}\\\\[0.5cm]

        \\cos(\Delta) = \\min( 1.0, \quad  \\sin( A_{\\mathrm{lat'}}) \
                     \\sin( B_{\\mathrm{lat'}} ) + \
                     \\cos(A_{\\mathrm{lat'}})  \\cos( B_{\\mathrm{lat'}} ) \
                     \\cos( B_{\\mathrm{lon'}} - A_{\\mathrm{lon'}} )

    :param a: Location point A
    :type a: :py:class:`pyrocko.orthodrome.Loc`
    :param b: Location point B
    :type b: :py:class:`pyrocko.orthodrome.Loc`

    :return: cosdelta
    :rtype: float
    '''

    alat, alon, blat, blon = _latlon_pair(args)

    return min(
        1.0,
        math.sin(alat*d2r) * math.sin(blat*d2r) +
        math.cos(alat*d2r) * math.cos(blat*d2r) *
        math.cos(d2r*(blon-alon)))


def cosdelta_numpy(a_lats, a_lons, b_lats, b_lons):
    '''Cosine of the angular distance between two points ``a`` and ``b``
    on a sphere.

    This function returns the cosines of the distance
    angles *delta* between two points ``a`` and ``b`` given as
    :py:class:`numpy.ndarray`.
    The coordinates are expected to be given in geographical coordinates
    and in degrees. For numerical stability a maximum of ``1.0`` is enforced.

    Please find the details of the implementation in the documentation of
    the function :py:func:`pyrocko.orthodrome.cosdelta` above.

    :param a_lats: Latitudes (degree) point A
    :param a_lons: Longitudes (degree) point A
    :param b_lats: Latitudes (degree) point B
    :param b_lons: Longitudes (degree) point B
    :type a_lats: :py:class:`numpy.ndarray`
    :type a_lons: :py:class:`numpy.ndarray`
    :type b_lats: :py:class:`numpy.ndarray`
    :type b_lons: :py:class:`numpy.ndarray`

    :return: cosdelta
    :type b_lons: :py:class:`numpy.ndarray`, ``(N)``
    '''
    return num.minimum(
        1.0,
        num.sin(a_lats*d2r) * num.sin(b_lats*d2r) +
        num.cos(a_lats*d2r) * num.cos(b_lats*d2r) *
        num.cos(d2r*(b_lons-a_lons)))


def azimuth(*args):
    '''Azimuth calculation

    This function (find implementation below) returns azimuth ...
    between points ``a`` and ``b``, coordinates of
    which are expected to be given in geographical coordinates and in degrees.

    .. math::

        A_{\\mathrm{lat'}} = \\frac{ \pi}{180} \cdot A_{lat}, \quad \
        A_{\\mathrm{lon'}} = \\frac{ \pi}{180} \cdot A_{lon}, \quad \
        B_{\\mathrm{lat'}} = \\frac{ \pi}{180} \cdot B_{lat}, \quad \
        B_{\\mathrm{lon'}} = \\frac{ \pi}{180} \cdot B_{lon}\\\\

        \\varphi_{\\mathrm{azi},AB} = \\frac{180}{\pi} \\arctan \left[ \\frac{\
                \\cos( A_{\\mathrm{lat'}}) \\cos( B_{\\mathrm{lat'}} ) \
                \\sin(B_{\\mathrm{lon'}} - A_{\\mathrm{lon'}} )} \
                {\\sin ( B_{\\mathrm{lat'}} ) - \\sin( A_{\\mathrm{lat'}} \
                  cosdelta) } \\right]

    :param a: Location point A
    :type a: :py:class:`pyrocko.orthodrome.Loc`
    :param b: Location point B
    :type b: :py:class:`pyrocko.orthodrome.Loc`

    :return: Azimuth in degree
    '''

    alat, alon, blat, blon = _latlon_pair(args)

    return r2d*math.atan2(
        math.cos(alat*d2r) * math.cos(blat*d2r) *
        math.sin(d2r*(blon-alon)),
        math.sin(d2r*blat) - math.sin(d2r*alat) * cosdelta(
            alat, alon, blat, blon))


def azimuth_numpy(a_lats, a_lons, b_lats, b_lons, _cosdelta=None):
    '''Calculation of the azimuth (*track angle*) from a location A towards B.

    This function returns azimuths (*track angles*) from locations A towards B
    given in :py:class:`numpy.ndarray`. Coordinates are expected to be given in
    geographical coordinates and in degrees.

    Please find the details of the implementation in the documentation of the
    function :py:func:`pyrocko.orthodrome.azimuth`.


    :param a_lats: Latitudes (degree) point A
    :param a_lons: Longitudes (degree) point A
    :param b_lats: Latitudes (degree) point B
    :param b_lons: Longitudes (degree) point B
    :type a_lats: :py:class:`numpy.ndarray`, ``(N)``
    :type a_lons: :py:class:`numpy.ndarray`, ``(N)``
    :type b_lats: :py:class:`numpy.ndarray`, ``(N)``
    :type b_lons: :py:class:`numpy.ndarray`, ``(N)``

    :return: Azimuths in degrees
    :rtype: :py:class:`numpy.ndarray`, ``(N)``
    '''
    if _cosdelta is None:
        _cosdelta = cosdelta_numpy(a_lats, a_lons, b_lats, b_lons)

    return r2d*num.arctan2(
        num.cos(a_lats*d2r) * num.cos(b_lats*d2r) *
        num.sin(d2r*(b_lons-a_lons)),
        num.sin(d2r*b_lats) - num.sin(d2r*a_lats) * _cosdelta)


def azibazi(*args):
    alat, alon, blat, blon = _latlon_pair(args)
    if alat == blat and alon == blon:
        return 0., 180.

    cd = cosdelta(alat, alon, blat, blon)
    azi = r2d*math.atan2(
        math.cos(alat*d2r) * math.cos(blat*d2r) *
        math.sin(d2r*(blon-alon)),
        math.sin(d2r*blat) - math.sin(d2r*alat) * cd)
    bazi = r2d*math.atan2(
        math.cos(blat*d2r) * math.cos(alat*d2r) *
        math.sin(d2r*(alon-blon)),
        math.sin(d2r*alat) - math.sin(d2r*blat) * cd)

    return azi, bazi


def azibazi_numpy(a_lats, a_lons, b_lats, b_lons):
    _cosdelta = cosdelta_numpy(a_lats, a_lons, b_lats, b_lons)
    azis = azimuth_numpy(a_lats, a_lons, b_lats, b_lons, _cosdelta)
    bazis = azimuth_numpy(b_lats, b_lons, a_lats, a_lons, _cosdelta)
    return azis, bazis


def azidist_numpy(*args):
    '''Calculation of the azimuth (*track angle*) and the distance from
    locations A towards B on a sphere.

    The assisting functions used are :py:func:`pyrocko.orthodrome.cosdelta` and
    :py:func:`pyrocko.orthodrome.azimuth`

    :param a_lats: Latitudes (degree) point A
    :param a_lons: Longitudes (degree) point A
    :param b_lats: Latitudes (degree) point B
    :param b_lons: Longitudes (degree) point B
    :type a_lats: :py:class:`numpy.ndarray`, ``(N)``
    :type a_lons: :py:class:`numpy.ndarray`, ``(N)``
    :type b_lats: :py:class:`numpy.ndarray`, ``(N)``
    :type b_lons: :py:class:`numpy.ndarray`, ``(N)``

    :return: Azimuths in degrees, distances in degrees
    :rtype: :py:class:`numpy.ndarray`, ``(2xN)``
    '''
    _cosdelta = cosdelta_numpy(*args)
    _azimuths = azimuth_numpy(_cosdelta=_cosdelta, *args)
    return _azimuths, r2d*num.arccos(_cosdelta)


def distance_accurate50m(*args):
    ''' Accurate distance calculation based on a spheroid of rotation.

    Function returns distance in meter between points A and B, coordinates of
    which must be given in geographical coordinates and in degrees.
    The returned distance should be accurate to 50\,m using WGS84.
    Values for the Earth's equator radius and the Earth's oblateness
    (``f_oblate``) are defined in the pyrocko configuration file
    :py:class:`pyrocko.config`.

    From wikipedia (http://de.wikipedia.org/wiki/Orthodrome), based on:

    ``Meeus, J.: Astronomical Algorithms, S 85, Willmann-Bell,
    Richmond 2000 (2nd ed., 2nd printing), ISBN 0-943396-61-1``

    .. math::

        F = \\frac{\pi}{180} \
                            \\frac{(A_{lat} + B_{lat})}{2}, \quad \
        G = \\frac{\pi}{180}  \
                            \\frac{(A_{lat} - B_{lat})}{2}, \quad \
        l = \\frac{\pi}{180} \
                            \\frac{(A_{lon} - B_{lon})}{2} \quad \
        \\\\[0.5cm]
        S = \\sin^2(G) \\cdot \\cos^2(l) + \
              \\cos^2(F) \\cdot \\sin^2(l), \quad \quad
        C = \\cos^2(G) \\cdot \\cos^2(l) + \
                  \\sin^2(F) \\cdot \\sin^2(l)

    .. math::

        w = \\arctan \\left( \\sqrt{ \\frac{S}{C}} \\right) , \quad \
        r =  \\sqrt{\\frac{S}{C} }

    The spherical-earth distance D between A and B, can be given with:

    .. math::

        D_{sphere} = 2w \cdot R_{equator}

    The oblateness of the Earth requires some correction with
    correction factors h1 and h2:

     .. math::

        h_1 = \\frac{3r - 1}{2C}, \quad \
        h_2 = \\frac{3r +1 }{2S}\\\\[0.5cm]

        D = D_{\\mathrm{sphere}} \cdot [ 1 + h_1 \,f_{\\mathrm{oblate}} \
                 \cdot \\sin^2(F) \
                 \\cos^2(G) - h_2\, f_{\\mathrm{oblate}} \
                 \cdot \\cos^2(F) \\sin^2(G)]


    :param a: Location point A
    :type a: :py:class:`pyrocko.orthodrome.Loc`
    :param b: Location point B
    :type b: :py:class:`pyrocko.orthodrome.Loc`

    :return: Distance in meter
    :rtype: float
    '''

    alat, alon, blat, blon = _latlon_pair(args)

    f = (alat + blat)*d2r / 2.
    g = (alat - blat)*d2r / 2.
    l = (alon - blon)*d2r / 2.

    s = math.sin(g)**2 * math.cos(l)**2 + math.cos(f)**2 * math.sin(l)**2
    c = math.cos(g)**2 * math.cos(l)**2 + math.sin(f)**2 * math.sin(l)**2

    w = math.atan(math.sqrt(s/c))

    if w == 0.0:
        return 0.0

    r = math.sqrt(s*c)/w
    d = 2.*w*earthradius_equator
    h1 = (3.*r-1.)/(2.*c)
    h2 = (3.*r+1.)/(2.*s)

    return d * (1. +
                earth_oblateness * h1 * math.sin(f)**2 * math.cos(g)**2 -
                earth_oblateness * h2 * math.cos(f)**2 * math.sin(g)**2)


def distance_accurate50m_numpy(a_lats, a_lons, b_lats, b_lons):
    ''' Accurate distance calculation based on a spheroid of rotation.

    Function returns distance in meter between points ``a`` and ``b``,
    coordinates of which must be given in geographical coordinates and in
    degrees.
    The returned distance should be accurate to 50\,m using WGS84.
    Values for the Earth's equator radius and the Earth's oblateness
    (``f_oblate``) are defined in the pyrocko configuration file
    :py:class:`pyrocko.config`.

    From wikipedia (http://de.wikipedia.org/wiki/Orthodrome), based on:

    ``Meeus, J.: Astronomical Algorithms, S 85, Willmann-Bell,
    Richmond 2000 (2nd ed., 2nd printing), ISBN 0-943396-61-1``

    .. math::

        F_i = \\frac{\pi}{180} \
                           \\frac{(a_{lat,i} + a_{lat,i})}{2}, \quad \
        G_i = \\frac{\pi}{180}  \
                           \\frac{(a_{lat,i} - b_{lat,i})}{2}, \quad \
        l_i= \\frac{\pi}{180} \
                            \\frac{(a_{lon,i} - b_{lon,i})}{2} \\\\[0.5cm]
        S_i = \\sin^2(G_i) \\cdot \\cos^2(l_i) + \
              \\cos^2(F_i) \\cdot \\sin^2(l_i), \quad \quad
        C_i = \\cos^2(G_i) \\cdot \\cos^2(l_i) + \
                  \\sin^2(F_i) \\cdot \\sin^2(l_i)

    .. math::

        w_i = \\arctan \\left( \\sqrt{\\frac{S_i}{C_i}} \\right), \quad \
        r_i =  \\sqrt{\\frac{S_i}{C_i} }

    The spherical-earth distance ``D`` between ``a`` and ``b``,
    can be given with:

    .. math::

        D_{\\mathrm{sphere},i} = 2w_i \cdot R_{\\mathrm{equator}}

    The oblateness of the Earth requires some correction with
    correction factors ``h1`` and ``h2``:

     .. math::

        h_{1.i} = \\frac{3r - 1}{2C_i}, \quad \
        h_{2,i} = \\frac{3r +1 }{2S_i}\\\\[0.5cm]

        D_{AB,i} = D_{\\mathrm{sphere},i} \cdot [1 + h_{1,i}
            \,f_{\\mathrm{oblate}} \
            \cdot \\sin^2(F_i) \
            \\cos^2(G_i) - h_{2,i}\, f_{\\mathrm{oblate}} \
            \cdot \\cos^2(F_i) \\sin^2(G_i)]

    :param a_lats: Latitudes (degree) point A
    :param a_lons: Longitudes (degree) point A
    :param b_lats: Latitudes (degree) point B
    :param b_lons: Longitudes (degree) point B
    :type a_lats: :py:class:`numpy.ndarray`, ``(N)``
    :type a_lons: :py:class:`numpy.ndarray`, ``(N)``
    :type b_lats: :py:class:`numpy.ndarray`, ``(N)``
    :type b_lons: :py:class:`numpy.ndarray`, ``(N)``

    :return: Distances in degrees
    :rtype: :py:class:`numpy.ndarray`, ``(N)``
    '''

    eq = num.logical_and(a_lats == b_lats, a_lons == b_lons)
    ii_neq = num.where(num.logical_not(eq))[0]

    if num.all(eq):
        return num.zeros_like(eq, dtype=num.float)

    def extr(x):
        if isinstance(x, num.ndarray) and x.size > 1:
            return x[ii_neq]
        else:
            return x

    a_lats = extr(a_lats)
    a_lons = extr(a_lons)
    b_lats = extr(b_lats)
    b_lons = extr(b_lons)

    f = (a_lats + b_lats)*d2r / 2.
    g = (a_lats - b_lats)*d2r / 2.
    l = (a_lons - b_lons)*d2r / 2.

    s = num.sin(g)**2 * num.cos(l)**2 + num.cos(f)**2 * num.sin(l)**2
    c = num.cos(g)**2 * num.cos(l)**2 + num.sin(f)**2 * num.sin(l)**2

    w = num.arctan(num.sqrt(s/c))

    r = num.sqrt(s*c)/w

    d = 2.*w*earthradius_equator
    h1 = (3.*r-1.)/(2.*c)
    h2 = (3.*r+1.)/(2.*s)

    dists = num.zeros(eq.size, dtype=num.float)
    dists[ii_neq] = d * (
        1. +
        earth_oblateness * h1 * num.sin(f)**2 * num.cos(g)**2 -
        earth_oblateness * h2 * num.cos(f)**2 * num.sin(g)**2)

    return dists


def ne_to_latlon(lat0, lon0, north_m, east_m):
    '''Transform local cartesian coordinates to latitude and longitude.

    From east and north coordinates (``x`` and ``y`` coordinate
    :py:class:`numpy.ndarray`)  relative to a reference differences in
    longitude and latitude are calculated, which are effectively changes in
    azimuth and distance, respectively:

    .. math::

        \\text{distance change:}\; \Delta {\\bf{a}} &= \sqrt{{\\bf{y}}^2 + \
                                           {\\bf{x}}^2 }/ \mathrm{R_E},

        \\text{azimuth change:}\; \Delta \\bf{\gamma} &= \\arctan( \\bf{x}  \
                                                         / \\bf{y}).

    The projection used preserves the azimuths of the input points.

    :param lat0: Latitude origin of the cartesian coordinate system.
    :param lon0: Longitude origin of the cartesian coordinate system.
    :param north_m: Northing distances from origin in meters.
    :param east_m: Easting distances from origin in meters.
    :type north_m: :py:class:`numpy.ndarray`, ``(N)``
    :type east_m: :py:class:`numpy.ndarray`, ``(N)``
    :type lat0: float
    :type lon0: float

    :return: Array with latitudes and longitudes
    :rtype: :py:class:`numpy.ndarray`, ``(2xN)``

    '''

    a = num.sqrt(north_m**2+east_m**2)/config().earthradius
    gamma = num.arctan2(east_m, north_m)

    return azidist_to_latlon_rad(lat0, lon0, gamma, a)


def azidist_to_latlon(lat0, lon0, azimuth_deg, distance_deg):
    '''(Durchreichen??).

    '''

    return azidist_to_latlon_rad(
        lat0, lon0, azimuth_deg/180.*num.pi, distance_deg/180.*num.pi)


def azidist_to_latlon_rad(lat0, lon0, azimuth_rad, distance_rad):
    ''' Absolute latitudes and longitudes are calculated from relative changes.

    For numerical stability a range between of ``-1.0`` and ``1.0`` is
    enforced for ``c`` and ``alpha``.

    .. math::

        \Delta {\\bf a}_i \; \\text{and} \; \Delta \gamma_i \; \
            \\text{are relative distances and azimuths from lat0 and lon0 for \
                \\textit{i} source points of a finite source.}

    .. math::

        \mathrm{b} &= \\frac{\pi}{2} -\\frac{\pi}{180} \;\mathrm{lat_0}\\\\
        {\\bf c}_i &=\\arccos[\; \cos(\Delta {\\bf{a}}_i) \cos(\mathrm{b}) \
              + |\Delta \gamma_i| \,\sin(\Delta {\\bf a}_i)
                \sin(\mathrm{b})\; ] \\\\
        \mathrm{lat}_i &=  \\frac{180}{\pi}  \
                          \\left(\\frac{\pi}{2} - {\\bf c}_i \\right)

    .. math::

         \\alpha_i &= \\arcsin \\left[ \; \\frac{ \sin(\Delta {\\bf a}_i ) \
                         \sin(|\Delta \\gamma_i|)}{\sin({\\bf c}_i)}\;
                         \\right] \\\\
        \\alpha_i &= \\begin{cases}
                 \\alpha_i, &\\text{if}  \; \cos(\Delta {\\bf a}_i) - \
                 \cos(\mathrm{b}) \cos({\\bf{c}}_i) > 0, \; \
                 \\text{else} \\\\
                 \pi - \\alpha_i, & \\text{if} \; \\alpha_i > 0,\;
                 \\text{else}\\\\
                -\pi - \\alpha_i, & \\text{if} \; \\alpha_i < 0.
                \\end{cases} \\\\
        \mathrm{lon}_i &=   \mathrm{lon_0} + \
             \\frac{180}{\pi} \, \\frac{\Delta \gamma_i }{|\Delta \gamma_i|} \
                            \cdot \\alpha_i  \
                             \\text{, with $\\alpha_i \\in [-\pi,\pi]$}

    :param lat0: Latitude origin of the cartesian coordinate system.
    :param lon0: Longitude origin of the cartesian coordinate system.
    :param distance_rad: Distances from origin in radians.
    :param azimuth_rad: Azimuth from radians.
    :type distance_rad: :py:class:`numpy.ndarray`, ``(N)``
    :type azimuth_rad: :py:class:`numpy.ndarray`, ``(N)``
    :type lat0: float
    :type lon0: float

    :return: Array with latitudes and longitudes
    :rtype: :py:class:`numpy.ndarray`, ``(2xN)``
    '''

    a = distance_rad
    gamma = azimuth_rad

    b = math.pi/2.-lat0*d2r

    alphasign = 1.
    alphasign = num.where(gamma < 0, -1., 1.)
    gamma = num.abs(gamma)

    c = num.arccos(clip(
        num.cos(a)*num.cos(b)+num.sin(a)*num.sin(b)*num.cos(gamma), -1., 1.))

    alpha = num.arcsin(clip(
        num.sin(a)*num.sin(gamma)/num.sin(c), -1., 1.))

    alpha = num.where(
        num.cos(a)-num.cos(b)*num.cos(c) < 0,
        num.where(alpha > 0,  math.pi-alpha, -math.pi-alpha),
        alpha)

    lat = r2d * (math.pi/2. - c)
    lon = wrap(lon0 + r2d*alpha*alphasign, -180., 180.)

    return lat, lon


def ne_to_latlon_alternative_method(lat0, lon0, north_m, east_m):
    '''Transform local cartesian coordinates to latitude and longitude.

    Like :py:func:`pyrocko.orthodrome.ne_to_latlon`,
    but this method (implementation below), although it should be numerically
    more stable, suffers problems at points which are *across the pole*
    as seen from the cartesian origin.

    .. math::

        \\text{distance change:}\; \Delta {{\\bf a}_i} &=\sqrt{{\\bf{y}}^2_i +
                                        {\\bf{x}}^2_i }/ \mathrm{R_E},\\\\
        \\text{azimuth change:}\; \Delta {\\bf \gamma}_i &=
                                        \\arctan( {\\bf x}_i {\\bf y}_i). \\\\
        \mathrm{b} &= \\frac{\pi}{2} -\\frac{\pi}{180} \;\mathrm{lat_0}\\\\

    .. math::

        {{\\bf z}_1}_i &= \\cos{\\left( \\frac{\Delta {\\bf a}_i - \
                            \\mathrm{b}}{2} \\right)} \
                            \\cos {\\left( \\frac{|\gamma_i|}{2} \\right) }\\\\
        {{\\bf n}_1}_i &= \\cos{\\left( \\frac{\Delta {\\bf a}_i + \
                            \\mathrm{b}}{2} \\right)}\
                            \\sin {\\left( \\frac{|\gamma_i|}{2} \\right) }\\\\
        {{\\bf z}_2}_i &= \\sin{\\left( \\frac{\Delta {\\bf a}_i - \
                            \\mathrm{b}}{2} \\right)}\
                            \\cos {\\left( \\frac{|\gamma_i|}{2} \\right) }\\\\
        {{\\bf n}_2}_i &= \\sin{\\left( \\frac{\Delta {\\bf a}_i + \
                            \\mathrm{b}}{2} \\right)}\
                            \\sin {\\left( \\frac{|\gamma_i|}{2} \\right) }\\\\
        {{\\bf t}_1}_i &= \\arctan{\\left( \\frac{{{\\bf z}_1}_i} \
                                    {{{\\bf n}_1}_i} \\right) }\\\\
        {{\\bf t}_2}_i &= \\arctan{\\left( \\frac{{{\\bf z}_2}_i} \
                                    {{{\\bf n}_2}_i} \\right) } \\\\[0.5cm]
        c &= \\begin{cases}
              2 \cdot \\arccos \\left( {{\\bf z}_1}_i / \\sin({{\\bf t}_1}_i) \
                              \\right),\; \\text{if } \
                              |\\sin({{\\bf t}_1}_i)| > \
                                |\\sin({{\\bf t}_2}_i)|,\; \\text{else} \\\\
              2 \cdot \\arcsin{\\left( {{\\bf z}_2}_i / \
                                 \\sin({{\\bf t}_2}_i) \\right)}.
             \\end{cases}\\\\

    .. math::

        {\\bf {lat}}_i  &= \\frac{180}{ \pi } \\left( \\frac{\pi}{2} \
                                              - {\\bf {c}}_i \\right) \\\\
        {\\bf {lon}}_i &=  {\\bf {lon}}_0 + \\frac{180}{ \pi } \
                                      \\frac{\gamma_i}{|\gamma_i|}, \
                                     \\text{ with}\; \gamma \\in [-\pi,\pi]

    :param lat0: Latitude origin of the cartesian coordinate system.
    :param lon0: Longitude origin of the cartesian coordinate system.
    :param north_m: Northing distances from origin in meters.
    :param east_m: Easting distances from origin in meters.
    :type north_m: :py:class:`numpy.ndarray`, ``(N)``
    :type east_m: :py:class:`numpy.ndarray`, ``(N)``
    :type lat0: float
    :type lon0: float

    :return: Array with latitudes and longitudes
    :rtype: :py:class:`numpy.ndarray`, ``(2xN)``
    '''

    b = math.pi/2.-lat0*d2r
    a = num.sqrt(north_m**2+east_m**2)/config().earthradius

    gamma = num.arctan2(east_m, north_m)
    alphasign = 1.
    alphasign = num.where(gamma < 0., -1., 1.)
    gamma = num.abs(gamma)

    z1 = num.cos((a-b)/2.)*num.cos(gamma/2.)
    n1 = num.cos((a+b)/2.)*num.sin(gamma/2.)
    z2 = num.sin((a-b)/2.)*num.cos(gamma/2.)
    n2 = num.sin((a+b)/2.)*num.sin(gamma/2.)
    t1 = num.arctan2(z1, n1)
    t2 = num.arctan2(z2, n2)

    alpha = t1 + t2

    sin_t1 = num.sin(t1)
    sin_t2 = num.sin(t2)
    c = num.where(
        num.abs(sin_t1) > num.abs(sin_t2),
        num.arccos(z1/sin_t1)*2.,
        num.arcsin(z2/sin_t2)*2.)

    lat = r2d * (math.pi/2. - c)
    lon = wrap(lon0 + r2d*alpha*alphasign, -180., 180.)
    return lat, lon


def latlon_to_ne(*args):
    '''Relative cartesian coordinates with respect to a reference location.

    For two locations, a reference location A and another location B, given in
    geographical coordinates in degrees, the corresponding cartesian
    coordinates are calculated.
    Assisting functions are :py:func:`pyrocko.orthodrome.azimuth' and
    :py:func:`pyrocko.orthodrome.distance_accurate50m`

    .. math::

        D_{AB} &= \\mathrm{distance\\_accurate50m(}A, B \\mathrm{)}, \quad \
                        \\varphi_{\\mathrm{azi},AB} = \\mathrm{azimuth(}A,B \
                            \\mathrm{)}\\\\[0.3cm]

        n &= D_{AB} \cdot \\cos( \\frac{\pi }{180} \
                                    \\varphi_{\\mathrm{azi},AB} )\\\\
        e &= D_{AB} \cdot \\sin( \\frac{\pi }{180} \\varphi_{\\mathrm{azi},AB})

    :param refloc: Location reference point
    :type refloc: :py:class:`pyrocko.orthodrome.Loc`
    :param loc: Location of interest
    :type loc: :py:class:`pyrocko.orthodrome.Loc`

    :return: Northing and easting from refloc to location
    :rtype: tuple, float

    '''

    azi = azimuth(*args)
    dist = distance_accurate50m(*args)
    n, e = math.cos(azi*d2r)*dist, math.sin(azi*d2r)*dist
    return n, e


def latlon_to_ne_numpy(lat0, lon0, lat, lon):
    '''Relative cartesian coordinates with respect to a reference location.

    For two locations, a reference location (``lat0``, ``lon0``) and another
    location B, given in geographical coordinates in degrees,
    the corresponding cartesian coordinates are calculated.
    Assisting functions are :py:func:`azimuth`
    and :py:func:`distance_accurate50m`.

    :param lat0: reference location latitude
    :param lon0: reference location longitude
    :param lat: absolute location latitude
    :param lon: absolute location longitude

    :return: ``(n, e)``: relative north and east positions
    :rtype: :py:class:`numpy.ndarray`, ``(2xN)``

    Implemented formulations:

       .. math::

           D_{AB} &= \\mathrm{distance\\_accurate50m(}A, B \\mathrm{)}, \quad \
           \\varphi_{\\mathrm{azi},AB} = \\mathrm{azimuth(}A,B \
                                                \\mathrm{)}\\\\[0.3cm]

           n &= D_{AB} \cdot \\cos( \\frac{\pi }{180} \\varphi_{ \
                                                \\mathrm{azi},AB} )\\\\
           e &= D_{AB} \cdot \\sin( \\frac{\pi }{180} \\varphi_{ \
                                                \\mathrm{azi},AB} )
    '''

    azi = azimuth_numpy(lat0, lon0, lat, lon)
    dist = distance_accurate50m_numpy(lat0, lon0, lat, lon)
    n = num.cos(azi*d2r)*dist
    e = num.sin(azi*d2r)*dist
    return n, e


_wgs84 = None


def get_wgs84():
    global _wgs84
    if _wgs84 is None:
        from geographiclib.geodesic import Geodesic
        _wgs84 = Geodesic.WGS84

    return _wgs84


def amap(n):
    def wrap(f):
        if n == 1:
            def func(*args):
                it = num.nditer(args + (None,))
                for ops in it:
                    ops[-1][...] = f(*ops[:-1])

                return it.operands[-1]
        elif n == 2:
            def func(*args):
                it = num.nditer(args + (None, None))
                for ops in it:
                    ops[-2][...], ops[-1][...] = f(*ops[:-2])

                return it.operands[-2], it.operands[-1]

        return func
    return wrap


@amap(2)
def ne_to_latlon2(lat0, lon0, north_m, east_m):
    wgs84 = get_wgs84()
    az = num.arctan2(east_m, north_m)*r2d
    dist = num.sqrt(east_m**2 + north_m**2)
    x = wgs84.Direct(lat0, lon0, az, dist)
    return x['lat2'], x['lon2']


@amap(2)
def latlon_to_ne2(lat0, lon0, lat1, lon1):
    wgs84 = get_wgs84()
    x = wgs84.Inverse(lat0, lon0, lat1, lon1)
    dist = x['s12']
    az = x['azi1']
    n = num.cos(az*d2r)*dist
    e = num.sin(az*d2r)*dist
    return n, e


@amap(1)
def distance_accurate15nm(lat1, lon1, lat2, lon2):
    wgs84 = get_wgs84()
    return wgs84.Inverse(lat1, lon1, lat2, lon2)['s12']


def positive_region(region):
    '''Normalize parameterization of a rectangular geographical region.

    :param region: ``(west, east, south, north)``
    :returns: ``(west, east, south, north)``, where ``west <= east`` and
        where ``west`` and ``east`` are in the range ``[-180., 180.+360.[``
    '''
    west, east, south, north = [float(x) for x in region]

    assert -180. - 360. <= west < 180.
    assert -180. < east <= 180. + 360.
    assert -90. <= south < 90.
    assert -90. < north <= 90.

    if east < west:
        east += 360.

    if west < -180.:
        west += 360.
        east += 360.

    return (west, east, south, north)


def points_in_region(p, region):
    '''
    Check what points are contained in a rectangular geographical region.

    :param p: NumPy array of shape ``(N, 2)`` where each row is a
        ``(lat, lon)`` pair [deg]
    :param region: ``(west, east, south, north)`` [deg]
    :returns: NumPy array of shape ``(N)``, type ``bool``
    '''

    w, e, s, n = positive_region(region)
    return num.logical_and(
        num.logical_and(s <= p[:, 0], p[:, 0] <= n),
        num.logical_or(
            num.logical_and(w <= p[:, 1], p[:, 1] <= e),
            num.logical_and(w-360. <= p[:, 1], p[:, 1] <= e-360.)))


def point_in_region(p, region):
    '''
    Check if a point is contained in a rectangular geographical region.

    :param p: ``(lat, lon)`` [deg]
    :param region: ``(west, east, south, north)`` [deg]
    :returns: ``bool``
    '''

    w, e, s, n = positive_region(region)
    return num.logical_and(
        num.logical_and(s <= p[0], p[0] <= n),
        num.logical_or(
            num.logical_and(w <= p[1], p[1] <= e),
            num.logical_and(w-360. <= p[1], p[1] <= e-360.)))


def radius_to_region(lat, lon, radius):
    '''
    Get a rectangular region which fully contains a given circular region.

    :param lat,lon: center of circular region [deg]
    :param radius: radius of circular region [m]
    :return: rectangular region as ``(east, west, south, north)`` [deg]
    '''
    radius_deg = radius * m2d
    if radius_deg < 45.:
        lat_min = max(-90., lat - radius_deg)
        lat_max = min(90., lat + radius_deg)
        absmaxlat = max(abs(lat_min), abs(lat_max))
        if absmaxlat > 89:
            lon_min = -180.
            lon_max = 180.
        else:
            lon_min = max(
                -180. - 360.,
                lon - radius_deg / math.cos(absmaxlat*d2r))
            lon_max = min(
                180. + 360.,
                lon + radius_deg / math.cos(absmaxlat*d2r))

        lon_min, lon_max, lat_min, lat_max = positive_region(
            (lon_min, lon_max, lat_min, lat_max))

        return lon_min, lon_max, lat_min, lat_max

    else:
        return None


def geographic_midpoint(lats, lons, weights=None):
    '''Calculate geographic midpoints by finding the center of gravity.

    This method suffers from instabilities if points are centered around the
    poles.

    :param lats: array of latitudes
    :param lons: array of longitudes
    :param weights: array weighting factors (optional)
    :type lats: :py:class:`numpy.ndarray`, ``(N)``
    :type lons: :py:class:`numpy.ndarray`, ``(N)``
    :type weights: :py:class:`numpy.ndarray`, ``(N)``

    :return: Latitudes and longitudes of the modpoints
    :rtype: :py:class:`numpy.ndarray`, ``(2xN)``
    '''
    if not weights:
        weights = num.ones(len(lats))

    total_weigth = num.sum(weights)
    weights /= total_weigth
    lats = lats * d2r
    lons = lons * d2r
    x = num.sum(num.cos(lats) * num.cos(lons) * weights)
    y = num.sum(num.cos(lats) * num.sin(lons) * weights)
    z = num.sum(num.sin(lats) * weights)

    lon = num.arctan2(y, x)
    hyp = num.sqrt(x**2 + y**2)
    lat = num.arctan2(z, hyp)

    return lat/d2r, lon/d2r
