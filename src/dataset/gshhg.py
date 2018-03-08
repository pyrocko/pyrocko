# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import

import logging
import io
import struct
import time
import numpy as num

from os import path

from pyrocko import config, orthodrome

logger = logging.getLogger('pyrocko.dataset.gshhg')
config = config.config()

km = 1e3
micro_deg = 1e-6


class Polygon(object):
    '''Representation of a GSHHG polygon. '''
    RIVER_NOT_SET = 0

    LEVELS = ['LAND', 'LAKE', 'ISLAND_IN_LAKE', 'POND_IN_ISLAND_IN_LAKE',
              'ANTARCTIC_ICE_FRONT', 'ANTARCTIC_GROUNDING_LINE']

    SOURCE = ['CIA_WDBII', 'WVS', 'AC']

    def __init__(self, gshhg_file, offset, *attr):
        '''Initialise a GSHHG polygon

        :param gshhg_file: GSHHG binary file
        :type gshhg_file: str
        :param offset: This polygons' offset in binary file
        :type offset: int
        :param attr: Polygon attributes
             ``(pid, npoints, _flag, west, east, south, north,
              area, area_full, container, ancestor)``.
              See :file:`gshhg.h` for details.
        :type attr: tuple
        '''
        (self.pid, self.npoints, self._flag,
         self.west, self.east, self.south, self.north,
         self.area, self.area_full, self.container, self.ancestor) = attr

        self.west *= micro_deg
        self.east *= micro_deg
        self.south *= micro_deg
        self.north *= micro_deg

        self.level_no = (self._flag & 255)
        self.level = self.LEVELS[self.level_no - 1]
        self.version = (self._flag >> 8) & 255

        cross = (self._flag >> 16) & 3
        self.greenwhich_crossed = True if cross == 1 or cross == 3 else False
        self.dateline_crossed = True if cross == 2 or cross == 3 else False

        self.source = self.SOURCE[(self._flag >> 24) & 1]
        if self.level_no >= 5:
            self.source = self.SOURCE[2]

        self.river = (self._flag >> 25) & 1

        scale = 10.**(self._flag >> 26)
        self.area /= scale
        self.area_full /= scale

        self._points = None
        self._file = gshhg_file
        self._offset = offset

    @property
    def points(self):
        '''Points of the polygon as Nx2 as ``[:,[lat,lon]]``

        :rtype: :class:`numpy.ndarray`
        '''
        if self._points is None:
            with open(self._file) as db:
                db.seek(self._offset)
                self._points = num.fromfile(
                    db, dtype='>i4', count=self.npoints*2)\
                    .astype(num.float32)\
                    .reshape(self.npoints, 2)

            self._points = num.fliplr(self._points)
            if self.level_no in (2, 4):
                self._points = self._points[::-1, :]

            self._points *= micro_deg
        return self._points

    @property
    def lats(self):
        return self.points[:, 0]

    @property
    def lons(self):
        return self.points[:, 1]

    def _is_level(self, level):
        if self.level is self.LEVELS[level]:
            return True
        return False

    def is_land(self):
        ''' Check if the polygon is land.

        :rtype: bool
        '''
        return self._is_level(0)

    def is_lake(self):
        ''' Check if the polygon is a lake.

        :rtype: bool
        '''
        return self._is_level(1)

    def is_island_in_lake(self):
        ''' Check if the polygon is an island in a lake.

        :rtype: bool
        '''
        return self._is_level(2)

    def is_pond_in_island_in_lake(self):
        ''' Check if the polygon is pond on an island in a lake.

        :rtype: bool
        '''
        return self._is_level(3)

    def is_antarctic_icefront(self):
        ''' Check if the polygon antarctic icefront.

        :rtype: bool
        '''
        return self._is_level(4)

    def is_antarctic_grounding_line(self):
        ''' Check if the polygon is antarctic grounding line.
        :rtype: bool
        '''
        return self._is_level(5)

    def contains_point(self, point):
        ''' Check if the polygon contains a single `point`

        :param point: (lat, lon) of size 2
        :type point: tuple
        :rtype: bool
        '''
        if self.south <= point[0] <= self.north and\
           self.west <= point[1] <= self.east:
            return orthodrome.contains_point(self.points, point)
        return False

    def contains_points(self, points):
        ''' Check if the polygon contains a `points`

        :param points: Array of of size Nx2
        :type point: :class:`numpy.ndarray`
        :rtype: bool
        '''
        cond = num.all([self.south <= points[:, 0],
                        points[:, 0] <= self.north,
                        self.west <= points[:, 1],
                        points[:, 1] <= self.east],
                       axis=0)
        r = orthodrome.contains_points(self.points, points)
        logger.debug('%s: points inside %d' % (self.level, r.sum()))
        return r
        if num.any(cond):
            pass

        return num.zeros(points.shape[0], dtype=num.bool)

    def get_bounding_box(self):
        return (self.west, self.east, self.south, self.north)

    def __lt__(self, polygon):
        return self.level_no < polygon.level_no

    def __str__(self):
        rstr = '''Polygon id: {p.pid}
-------------------
Points:     {p.npoints}
Level:      {p.level}
Area:       {p.area} km**2
Area Full:  {p.area_full} km**2
Extent:     {p.west} W, {p.east} E, {p.south} S, {p.north} N
Source:     {p.source}
Greenwhich crossed: {p.greenwhich_crossed}
Dateline crossed:   {p.dateline_crossed}
        '''.format(p=self)
        return rstr


class GSHHG(object):
    '''Holding the Global Self-consistent Hierarchical High-resolutions
        Geography Database (GSHHG)

    The class offers methods to select :class:`~pyrocko.gshhg.Polygon` s and
    crop/validate single points and point-clouds.

    .. info:

        If the database is not available it will be downloaded and cached
        automatically

    .. note:

        Cite Wessel, P., and W. H. F. Smith, A Global Self-consistent,
        Hierarchical, High-resolution Shoreline Database,
        J. Geophys. Res., 101, #B4, pp. 8741-8743, 1996.
    '''

    gshhg_url = 'http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-bin-2.3.7.zip'
    _header_struct = struct.Struct('>IIIiiiiIIii')

    def __init__(self, gshhg_file):
        ''' Initialise the database from GSHHG binary.

        :param gshhg_file: Path to file
        :type gshhg_file: str
        :
        '''
        t0 = time.time()
        self._file = gshhg_file

        self.polygons = []
        self._read_database()
        logger.debug('Initialised GSHHG database from %s in [%.4f s]'
                     % (gshhg_file, time.time()-t0))

    def _read_database(self):
        with open(self._file, mode='rb') as db:
            while db:
                buf = db.read(self._header_struct.size)
                if not buf:
                    break
                header = self._header_struct.unpack_from(buf)
                p = Polygon(
                    self._file,
                    db.tell(),
                    *header)
                self.polygons.append(p)

                offset = 8 * header[1]
                db.seek(offset, io.SEEK_CUR)

    @classmethod
    def _get_database(cls, filename):
        file = path.join(config.gshhg_dir, filename)
        if not path.exists(file):
            from pyrocko import util
            import zipfile

            archive_path = path.join(config.gshhg_dir,
                                     path.basename(cls.gshhg_url))
            util.download_file(cls.gshhg_url, archive_path)
            if not zipfile.is_zipfile(archive_path):
                raise util.DownloadError('GSHHG file is corrupted!')
            logger.info('Unzipping GSHHG database...')
            zipf = zipfile.ZipFile(archive_path)
            zipf.extractall(config.gshhg_dir)
        else:
            logger.debug('Using cached %s' % filename)
        return file

    def get_polygons_at(self, lat, lon):
        '''Get all polygons that intersect with a point.

        :param lat: Latitude in [deg]
        :type lat: float
        :param lat: Longitude in [deg]
        :type lat: float
        :returns: List of :class:`~pyrocko.gshhg.Polygon`
        :rtype: list
        '''
        rp = []
        for p in self.polygons:
            if (p.west < lon and p.east > lon) and\
               (p.south < lat and p.north > lat):
                rp.append(p)
        return rp

    def get_polygons_within(self, west, east, south, north):
        '''Get all polygons that intersect with a bounding box.

        :param west: Western boundary in decimal degree
        :type west: float
        :param east: Eastern boundary in decimal degree
        :type east: float
        :param north: Northern boundary in decimal degree
        :type north: float
        :param south: Southern boundary in decimal degree
        :type south: float
        :returns: List of :class:`~pyrocko.gshhg.Polygon`
        :rtype: list
        '''
        rp = []
        for p in self.polygons:
            if ((p.west > west and p.east < east) or
               (p.west < west and p.east > west) or
               (p.west < east and p.east > east) or
               (p.west > west and p.east < east)) and\
               ((p.south > south and p.north < north) or
               (p.south < south and p.north > south) or
               (p.south < north and p.north > north) or
               (p.north > north and p.south < south)):
                rp.append(p)
        return rp

    def is_point_on_land(self, lat, lon):
        '''Check whether a point is on land. Consquently lakes are excluded.

        :param lat: Latitude in [deg]
        :type lat: float
        :param lon: Latitude in [deg]
        :type lon: float
        :rtype: bool
        '''
        relevant_polygons = self.get_polygons_at(lat, lon)
        relevant_polygons.sort()

        land = False
        for p in relevant_polygons:
            if (p.is_land() or p.is_antarctic_grounding_line() or
               p.is_island_in_lake()):
                land = True
            elif (p.is_lake() or p.is_antarctic_icefront() or
                  p.is_pond_in_island_in_lake()):
                land = False
        return land

    def get_land_mask(self, points):
        '''Get a landmask respecting lakes, and ponds in island in lake
            as water

        :param points: List of lat, lon pairs
        :type points: :class:`numpy.ndarray` of shape Nx2
        :return: Boolean land mask
        :rtype: :class:`numpy.ndarray` of shape N
        '''

        lats = points[:, 0]
        lons = points[:, 1]
        west, east, south, north = (lons.min(), lons.max(),
                                    lats.min(), lats.max())

        relevant_polygons = self.get_polygons_within(west, east, south, north)
        relevant_polygons.sort()

        mask = num.zeros(points.shape[0], dtype=num.bool)
        for p in relevant_polygons:
            if (p.is_land() or p.is_antarctic_grounding_line() or
               p.is_island_in_lake()):
                mask += p.contains_points(points)
            elif p.is_lake() or p.is_pond_in_island_in_lake():
                water = p.contains_points(points)
                num.logical_xor(mask, water, out=mask)
        return mask

    @classmethod
    def full(cls):
        ''' Return the full-resolution GSHHG database'''
        return cls(cls._get_database('gshhs_f.b'))

    @classmethod
    def high(cls):
        ''' Return the high-resolution GSHHG database'''
        return cls(cls._get_database('gshhs_h.b'))

    @classmethod
    def intermediate(cls):
        ''' Return the intermediate-resolution GSHHG database'''
        return cls(cls._get_database('gshhs_i.b'))

    @classmethod
    def low(cls):
        ''' Return the low-resolution GSHHG database'''
        return cls(cls._get_database('gshhs_l.b'))

    @classmethod
    def crude(cls):
        ''' Return the crude-resolution GSHHG database'''
        return cls(cls._get_database('gshhs_c.b'))
