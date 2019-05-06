import numpy as num
import math

from pyrocko import orthodrome, config
from pyrocko.guts import Object, Float


guts_prefix = 'pf'

d2r = math.pi / 180.
r2d = 1.0 / d2r
km = 1000.


def latlondepth_to_carthesian(lat, lon, depth):
    radius = config.config().earthradius - depth
    x = radius * math.cos(d2r*lat) * math.cos(d2r*lon)
    y = radius * math.cos(d2r*lat) * math.sin(d2r*lon)
    z = radius * math.sin(d2r*lat)
    return x, y, z


class Location(Object):
    '''
    Geographical location.

    The location is given by a reference point at the earth's surface
    (:py:attr:`lat`, :py:attr:`lon`) and a cartesian offset from this point
    (:py:attr:`north_shift`, :py:attr:`east_shift`, :py:attr:`depth`). The
    offset corrected lat/lon coordinates of the location can be accessed though
    the :py:attr:`effective_latlon`, :py:attr:`effective_lat`, and
    :py:attr:`effective_lon` properties.
    '''

    lat = Float.T(
        default=0.0,
        optional=True,
        help='latitude of reference point [deg]')

    lon = Float.T(
        default=0.0,
        optional=True,
        help='longitude of reference point [deg]')

    north_shift = Float.T(
        default=0.,
        optional=True,
        help='northward cartesian offset from reference point [m]')

    east_shift = Float.T(
        default=0.,
        optional=True,
        help='eastward cartesian offset from reference point [m]')

    elevation = Float.T(
        default=0.0,
        optional=True,
        help='elevation, above the surface [m]')

    depth = Float.T(
        default=0.0,
        help='depth, below the surface [m]')

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._latlon = None

    def __setattr__(self, name, value):
        if name in ('lat', 'lon', 'north_shift', 'east_shift'):
            self.__dict__['_latlon'] = None

        Object.__setattr__(self, name, value)

    @property
    def effective_latlon(self):
        '''
        Property holding the offset-corrected lat/lon pair of the location.
        '''

        if self._latlon is None:
            if self.north_shift == 0.0 and self.east_shift == 0.0:
                self._latlon = self.lat, self.lon
            else:
                self._latlon = tuple(float(x) for x in orthodrome.ne_to_latlon(
                    self.lat, self.lon, self.north_shift, self.east_shift))

        return self._latlon

    @property
    def effective_lat(self):
        '''
        Property holding the offset-corrected latitude of the location.
        '''

        return self.effective_latlon[0]

    @property
    def effective_lon(self):
        '''
        Property holding the offset-corrected longitude of the location.
        '''

        return self.effective_latlon[1]

    def same_origin(self, other):
        '''
        Check whether other location object has the same reference location.
        '''

        return self.lat == other.lat and self.lon == other.lon

    def distance_to(self, other):
        '''
        Compute surface distance [m] to other location object.


        '''

        if self.same_origin(other):
            if isinstance(other, Location):
                return math.sqrt((self.north_shift - other.north_shift)**2 +
                                 (self.east_shift - other.east_shift)**2)
            else:
                return 0.0

        else:
            slat, slon = self.effective_latlon
            try:
                rlat, rlon = other.effective_latlon
            except AttributeError:
                rlat, rlon = other.lat, other.lon

            return float(orthodrome.distance_accurate50m_numpy(
                slat, slon, rlat, rlon)[0])

    def distance_3d_to(self, other):
        '''
        Compute 3D distance [m] to other location object.

        All coordinates are transformed to cartesian coordinates if necessary
        then distance is:

        .. math::

            \\Delta = \\sqrt{\\Delta {\\bf x}^2 + \\Delta {\\bf y}^2 + \
                      \\Delta {\\bf z}^2}

        '''

        if self.same_origin(other):
            if isinstance(other, Location):
                return math.sqrt((self.north_shift - other.north_shift)**2 +
                                 (self.east_shift - other.east_shift)**2 +
                                 (self.depth - other.depth)**2)
            else:
                return 0.0
        else:
            slat, slon = self.effective_latlon
            try:
                rlat, rlon = other.effective_latlon
            except AttributeError:
                rlat, rlon = other.lat, other.lon

            sx, sy, sz = latlondepth_to_carthesian(slat, slon, self.depth)
            rx, ry, rz = latlondepth_to_carthesian(rlat, rlon, other.depth)

            return math.sqrt((sx-rx)**2 + (sy-ry)**2 + (sz-rz)**2)

    def azibazi_to(self, other):
        '''
        Compute azimuth and backazimuth to and from other location object.
        '''

        if self.same_origin(other):
            if isinstance(other, Location):
                azi = r2d * math.atan2(other.east_shift - self.east_shift,
                                       other.north_shift - self.north_shift)
            else:
                azi = 0.0

            bazi = azi + 180.
        else:
            slat, slon = self.effective_latlon
            try:
                rlat, rlon = other.effective_latlon
            except AttributeError:
                rlat, rlon = other.lat, other.lon

            azi, bazi = orthodrome.azibazi_numpy(slat, slon, rlat, rlon)

        return float(azi), float(bazi)

    def set_origin(self, lat, lon):
        lat = float(lat)
        lon = float(lon)
        elat, elon = self.effective_latlon
        n, e = orthodrome.latlon_to_ne_numpy(lat, lon, elat, elon)
        self.lat = lat
        self.lon = lon
        self.north_shift = float(n)
        self.east_shift = float(e)
        self._latlon = elat, elon  # unchanged

    @property
    def coords5(self):
        return num.array([
            self.lat, self.lon, self.north_shift, self.east_shift, self.depth])
