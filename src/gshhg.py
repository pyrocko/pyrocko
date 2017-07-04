from __future__ import absolute_import

import logging
import io
import struct
import numpy as num

from os import path

from . import config
from . import orthodrome

logger = logging.getLogger('pyrocko.gshhg')
config = config.config()

km = 1e3
micro_deg = 1e-6


class Polygon(object):
    RIVER_NOT_SET = 0

    LEVELS = ['LAND', 'LAKE', 'ISLAND_IN_LAKE', 'POND_IN_ISLAND_IN_LAKE',
              'ANTARCTIC_ICE_FRONT', 'ANTARCTIC_GROUNDING_LINE']

    SOURCE = ['CIA_WDBII', 'WVS', 'AC']

    def __init__(self, gshhg_file, offset, *args):
        (self.pid, self.npoints, self._flag,
         self.west, self.east, self.south, self.north,
         self.area, self.area_full, self.container, self.ancestor) = args

        self.west *= micro_deg
        self.east *= micro_deg
        self.south *= micro_deg
        self.north *= micro_deg

        self.level = self.LEVELS[(self._flag & 255) - 1]
        self.version = (self._flag >> 8) & 255

        cross = (self._flag >> 16) & 3
        self.greenwhich_crossed = True if cross == 1 or cross == 3 else False
        self.dateline_crossed = True if cross == 2 or cross == 3 else False

        self.source = self.SOURCE[(self._flag >> 24) & 1]
        if self.level >= 5:
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
        if self._points is None:
            with open(self._file) as db:
                db.seek(self._offset)
                self._points = num.fromfile(
                    db, dtype='>i4', count=self.npoints*2)\
                    .astype(num.float32)\
                    .reshape(self.npoints, 2)

                self._points = num.fliplr(self._points)
                self._points *= micro_deg
        return self._points

    @property
    def lats(self):
        return self.points[:, 1]

    @property
    def lons(self):
        return self.points[:, 0]

    def _is_level(self, level):
        if self.level == self.LEVELS[level]:
            return True
        return False

    def is_land(self):
        return self._is_level(0)

    def is_lake(self):
        return self._is_level(1)

    def is_island_in_lake(self):
        return self._is_level(2)

    def is_pond_in_island_in_lake(self):
        return self._is_level(3)

    def is_antarctic_icefront(self):
        return self._is_level(4)

    def is_antarctic_grounding_line(self):
        return self._is_level(5)

    def contains_point(self, point):
        if self.south <= point[0] <= self.north and\
           self.west <= point[1] <= self.east:
            return orthodrome.contains_point(self.points, point)
        return False

    def contains_points(self, points):
        cond = num.all([self.south <= points[:, 0],
                        points[:, 0] <= self.north,
                        self.west <= points[:, 1],
                        points[:, 1] <= self.east],
                       axis=0)

        if num.any(cond):
            return orthodrome.contains_points(self.points, points)
        return num.full(points.shape[1],
                        dtype=num.bool,
                        fill_value=False)

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

    gshhg_url = 'http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-bin-2.3.7.zip'

    _header_struct = struct.Struct('>IIIiiiiIIii')

    def __init__(self, gshhg_file):
        logger.debug('Initialising GSHHG database from %s' % gshhg_file)
        self._file = gshhg_file

        self.polygons = []
        self.read_database()

    def read_database(self):
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
            logger.info('Downloading GSHHG database (%s)...' % cls.gshhg_url)

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

    @classmethod
    def get_full(cls):
        return cls(cls._get_database('gshhs_f.b'))

    @classmethod
    def get_high(cls):
        return cls(cls._get_database('gshhs_h.b'))

    @classmethod
    def get_intermediate(cls):
        return cls(cls._get_database('gshhs_i.b'))

    @classmethod
    def get_crude(cls):
        return cls(cls._get_database('gshhs_c.b'))
