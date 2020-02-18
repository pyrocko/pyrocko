# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Slab2 subduction zone geometry model from the USGS (`Slab2
<https://doi.org/10.5066/F7PV6JNV>`_).

The model provides gridded depth, depth uncertainty, strike, dip and
thickness of the subducting slabs of 27 subduction zones on a regular
0.05 degree grid.

On first use, the Slab2 distribution archive is downloaded and the gridded
text files it contains are converted into a compact binary representation,
which is cached in the directory given by the ``slab2_dir`` setting of the
Pyrocko configuration (default: ``~/.pyrocko/slab2``).

Example::

    from pyrocko.dataset.slab2 import Slab2

    slabs = Slab2()
    slab = slabs.get_slab('sam')
    print(slab.name, slab.depth.shape)
    vertices, faces = slab.get_mesh()
'''

import os
import re
import logging
import tarfile
from os import path as op
from collections import OrderedDict

import numpy as num

from pyrocko import util, config, geometry
from .util import get_download_callback

logger = logging.getLogger('pyrocko.dataset.slab2')

km = 1e3

citation = '''
Hayes, G., 2018, Slab2 - A Comprehensive Subduction Zone Geometry Model:
U.S. Geological Survey data release, https://doi.org/10.5066/F7PV6JNV.
'''

__doc__ += '''

.. rubric:: Citation
''' + citation

prefix_to_slab_name = OrderedDict([
    ('alu', 'Aleutians'),
    ('cal', 'Calabria'),
    ('cam', 'Central America'),
    ('car', 'Caribbean'),
    ('cas', 'Cascadia'),
    ('cot', 'Cotabato'),
    ('hal', 'Halmahera'),
    ('hel', 'Hellenic'),
    ('him', 'Himalaya'),
    ('hin', 'Hindu Kush'),
    ('izu', 'Izu-Bonin'),
    ('ker', 'Kermadec-Tonga'),
    ('kur', 'Kamchatka-Kuril Islands-Japan'),
    ('mak', 'Makran'),
    ('man', 'Manila Trench'),
    ('mue', 'Muertos Trough'),
    ('pam', 'Pamir'),
    ('phi', 'Philippines'),
    ('png', 'New Guinea'),
    ('puy', 'Puysegur'),
    ('ryu', 'Ryukyu'),
    ('sam', 'South America'),
    ('sco', 'Scotia Sea'),
    ('sol', 'Solomon Islands'),
    ('sul', 'Sulawesi'),
    ('sum', 'Sumatra-Java'),
    ('van', 'Vanuatu'),
])

# Slab2 file name abbreviation -> (parameter name, factor to SI units)
#
# Slab2 depths are given in [km], negative downwards. Here, all depth-like
# parameters are converted to [m], with depth positive downwards.
abbr_to_parameter = OrderedDict([
    ('dep', ('depth', -km)),
    ('unc', ('depth_uncertainty', km)),
    ('str', ('strike', 1.0)),
    ('dip', ('dip', 1.0)),
    ('thk', ('thickness', km)),
])

parameter_units = {
    'depth': 'm',
    'depth_uncertainty': 'm',
    'strike': 'deg',
    'dip': 'deg',
    'thickness': 'm',
}

_re_xyz_member = re.compile(
    r'^(?P<prefix>[a-z]{3})_slab2_(?P<abbr>dep|unc|str|dip|thk)_'
    r'[0-9.]+\.xyz$')


def get_slab_names():
    '''
    Get full names of all Slab2 subduction zones.
    '''
    return list(prefix_to_slab_name.values())


def get_slab_prefixes():
    '''
    Get the three-letter abbreviations of all Slab2 subduction zones.
    '''
    return list(prefix_to_slab_name.keys())


def get_parameter_names():
    '''
    Get names of the parameters available for each slab.
    '''
    return [name for (name, _) in abbr_to_parameter.values()]


class Slab2Error(Exception):
    pass


def _xyz_to_grid(data, lats=None, lons=None):
    '''
    Convert Slab2 ``lon, lat, value`` triplets into a regular grid.

    If ``lats`` and ``lons`` are given, the values are put into that grid,
    otherwise, the grid coordinates are derived from the data.
    '''

    lon = data[:, 0]
    lat = data[:, 1]
    values = data[:, 2]

    if lats is None or lons is None:
        ulons = num.unique(lon)
        ulats = num.unique(lat)
        if ulons.size < 2 or ulats.size < 2:
            raise Slab2Error('Slab2 grid is too small.')

        dlon = num.min(num.diff(ulons))
        dlat = num.min(num.diff(ulats))
        nlon = int(round((ulons[-1] - ulons[0]) / dlon)) + 1
        nlat = int(round((ulats[-1] - ulats[0]) / dlat)) + 1
        lons = ulons[0] + num.arange(nlon) * dlon
        lats = ulats[0] + num.arange(nlat) * dlat

    dlon = lons[1] - lons[0]
    dlat = lats[1] - lats[0]

    ilon = num.round((lon - lons[0]) / dlon).astype(num.int64)
    ilat = num.round((lat - lats[0]) / dlat).astype(num.int64)

    mask = num.logical_and.reduce((
        ilon >= 0, ilon < lons.size,
        ilat >= 0, ilat < lats.size))

    grid = num.full((lats.size, lons.size), num.nan, dtype=num.float32)
    grid[ilat[mask], ilon[mask]] = values[mask]

    return lats, lons, grid


class Slab(object):
    '''
    Gridded geometry of a single subduction zone slab.

    Grid nodes outside of the slab are set to NaN.

    :ivar prefix: three-letter abbreviation of the slab
    :ivar name: full name of the subduction zone
    :ivar lats: latitudes of the grid nodes [deg], shape ``(nlat,)``
    :ivar lons: longitudes of the grid nodes [deg], shape ``(nlon,)``. As in
        the original dataset, these are in the range [0, 360].
    :ivar depth: depth of the slab surface [m], positive downwards
    :ivar depth_uncertainty: uncertainty of the slab depth [m]
    :ivar strike: strike of the slab surface [deg]
    :ivar dip: dip of the slab surface [deg]
    :ivar thickness: slab thickness [m]

    The gridded parameters are 2D arrays of shape ``(nlat, nlon)``.
    '''

    def __init__(self, prefix, lats, lons, **grids):
        self.prefix = prefix
        self.name = prefix_to_slab_name.get(prefix, prefix)
        self.lats = lats
        self.lons = lons
        for parameter in get_parameter_names():
            setattr(self, parameter, grids.get(parameter, None))

    def __str__(self):
        return 'Slab: %s (%s), lat: %g - %g, lon: %g - %g, ' \
            'depth: %g - %g km' % (
                self.name, self.prefix,
                self.lats.min(), self.lats.max(),
                self.lons.min(), self.lons.max(),
                num.nanmin(self.depth) / km, num.nanmax(self.depth) / km)

    @property
    def nlat(self):
        return self.lats.size

    @property
    def nlon(self):
        return self.lons.size

    def get_values(self, parameter):
        '''
        Get gridded values of a parameter.

        :param parameter: name of the parameter, see
            :py:func:`get_parameter_names`
        :returns: :py:class:`numpy.ndarray` of shape ``(nlat, nlon)``
        '''

        if parameter not in get_parameter_names():
            raise Slab2Error('Invalid parameter: %s' % parameter)

        values = getattr(self, parameter)
        if values is None:
            raise Slab2Error(
                'Parameter "%s" is not available for slab "%s".'
                % (parameter, self.prefix))

        return values

    def get_latlondepth(self):
        '''
        Get coordinates of all grid nodes inside the slab.

        :returns: :py:class:`numpy.ndarray` of shape ``(n, 3)``, with
            columns latitude [deg], longitude [deg] and depth [m]
        '''
        lons, lats = num.meshgrid(self.lons, self.lats)
        mask = num.isfinite(self.depth)
        return num.vstack((
            lats[mask], lons[mask], self.depth[mask])).T.astype(num.float64)

    def _get_mesh_indices(self):
        nlat, nlon = self.depth.shape
        faces = geometry.topo_to_faces_quad(nlat, nlon)
        valid = num.isfinite(self.depth.ravel())
        faces = faces[num.all(valid[faces], axis=1)]

        used = num.zeros(nlat * nlon, dtype=bool)
        used[faces.ravel()] = True
        ivertices = num.nonzero(used)[0]

        new_index = num.full(nlat * nlon, -1, dtype=num.int64)
        new_index[ivertices] = num.arange(ivertices.size)

        return ivertices, new_index[faces]

    def get_mesh(self, planetradius=None):
        '''
        Get quadrilateral surface mesh of the slab.

        Only cells for which all four corners are inside the slab are
        included.

        :param planetradius: planet radius [m], defaults to
            :py:data:`pyrocko.cake.earthradius`
        :returns: ``(vertices, faces)``, where ``vertices`` are Cartesian
            coordinates of shape ``(nvertices, 3)``, normalized to the planet
            radius, as used in :py:mod:`pyrocko.geometry`, and ``faces`` is
            an integer array of shape ``(nfaces, 4)`` with vertex indices.
        '''

        if planetradius is None:
            from pyrocko import cake
            planetradius = cake.earthradius

        ivertices, faces = self._get_mesh_indices()
        lons, lats = num.meshgrid(self.lons, self.lats)
        latlondepth = num.vstack((
            lats.ravel()[ivertices],
            lons.ravel()[ivertices],
            self.depth.ravel()[ivertices])).T.astype(num.float64)

        vertices = geometry.latlondepth2xyz(latlondepth, planetradius)
        return vertices, faces

    def get_face_values(self, parameter):
        '''
        Get values of a parameter for the faces of the mesh.

        The values are averaged over the four corners of each face, matching
        the faces returned by :py:meth:`get_mesh`.

        :param parameter: name of the parameter, see
            :py:func:`get_parameter_names`
        :returns: :py:class:`numpy.ndarray` of shape ``(nfaces,)``
        '''

        values = self.get_values(parameter).ravel().astype(num.float64)
        ivertices, faces = self._get_mesh_indices()
        corner_values = values[ivertices][faces]

        if parameter == 'strike':
            # circular mean
            a = corner_values * geometry.d2r
            return num.mod(num.arctan2(
                num.nanmean(num.sin(a), axis=1),
                num.nanmean(num.cos(a), axis=1)) * geometry.r2d, 360.)
        else:
            return num.nanmean(corner_values, axis=1)

    def get_geometry(self):
        '''
        Get slab surface as :py:class:`pyrocko.model.Geometry`.

        All available parameters are attached as face properties.

        :returns: :py:class:`pyrocko.model.Geometry` object
        '''

        from pyrocko.model import Geometry, Event
        from pyrocko.orthodrome import geographic_midpoint

        ivertices, faces = self._get_mesh_indices()
        lons, lats = num.meshgrid(self.lons, self.lats)

        vertices = num.zeros((ivertices.size, 5))
        vertices[:, 0] = lats.ravel()[ivertices]
        vertices[:, 1] = lons.ravel()[ivertices]
        vertices[:, 4] = self.depth.ravel()[ivertices]

        lat, lon = geographic_midpoint(vertices[:, 0], vertices[:, 1])
        event = Event(
            lat=float(lat), lon=float(lon),
            depth=float(num.mean(vertices[:, 4])),
            name='slab2_%s' % self.prefix)

        geom = Geometry(event=event)
        geom.setup(vertices, faces)
        for parameter in get_parameter_names():
            if getattr(self, parameter) is not None:
                geom.add_property(
                    parameter, self.get_face_values(parameter))

        return geom


class Slab2(object):
    '''
    Access to the Slab2 subduction zone geometry model.

    :param archive_path: path to a manually downloaded copy of the Slab2
        distribution archive (``Slab2Distribute_Mar2018.tar.gz``). If not
        given, the archive is expected in (or downloaded to) ``data_dir``.
    :param data_dir: directory where the converted data is cached. Defaults
        to the ``slab2_dir`` setting of the Pyrocko configuration.
    '''

    URL_SLAB2 = 'https://mirror.pyrocko.org/slab2/Slab2Distribute_Mar2018.tar.gz'  # noqa
    ARCHIVE_NAME = 'Slab2Distribute_Mar2018.tar.gz'

    def __init__(self, archive_path=None, data_dir=None):
        if data_dir is None:
            data_dir = config.config().slab2_dir

        self.data_dir = data_dir
        if archive_path is None:
            archive_path = op.join(self.data_dir, self.ARCHIVE_NAME)

        self.archive_path = archive_path
        self._slabs = {}
        self._converted = False

    @property
    def prefixes(self):
        return get_slab_prefixes()

    @property
    def names(self):
        return get_slab_names()

    def _cache_path(self, prefix):
        return op.join(self.data_dir, '%s.npz' % prefix)

    def download(self):
        '''
        Download the Slab2 distribution archive.
        '''

        logger.info('Downloading Slab2 geometry data...')
        util.ensuredirs(self.archive_path)
        try:
            util.download_file(
                self.URL_SLAB2, self.archive_path,
                status_callback=get_download_callback(
                    'Downloading Slab2 geometry data...'))

        except util.DownloadError as e:
            raise Slab2Error(
                'Download of Slab2 data failed: %s\n'
                'You may download the archive manually from %s and place it '
                'at %s' % (
                    str(e), 'https://doi.org/10.5066/F7PV6JNV',
                    self.archive_path))

    def convert(self):
        '''
        Convert gridded text files from the Slab2 archive to binary cache.

        All slabs are converted in a single pass through the archive.
        '''

        if not op.exists(self.archive_path):
            self.download()

        logger.info('Converting Slab2 data from archive: %s'
                    % self.archive_path)

        raw = {}
        with tarfile.open(self.archive_path, mode='r|gz') as tar:
            for member in tar:
                if not member.isfile():
                    continue

                m = _re_xyz_member.match(op.basename(member.name))
                if not m or m.group('prefix') not in prefix_to_slab_name:
                    continue

                logger.debug('Reading %s' % member.name)
                f = tar.extractfile(member)
                data = num.loadtxt(
                    f, delimiter=',', dtype=num.float64, ndmin=2)

                raw.setdefault(m.group('prefix'), {})[m.group('abbr')] = data

        if not raw:
            raise Slab2Error(
                'No Slab2 grid files found in archive: %s'
                % self.archive_path)

        for prefix, raw_slab in raw.items():
            if 'dep' not in raw_slab:
                logger.warning(
                    'No depth grid for slab "%s" found in archive, skipping.'
                    % prefix)
                continue

            lats, lons, _ = _xyz_to_grid(raw_slab['dep'])
            grids = {}
            for abbr, (parameter, factor) in abbr_to_parameter.items():
                if abbr in raw_slab:
                    _, _, grid = _xyz_to_grid(raw_slab[abbr], lats, lons)
                    grids[parameter] = grid * num.float32(factor)

            path = self._cache_path(prefix)
            util.ensuredirs(path)
            tmp_path = path + '.tmp.npz'
            num.savez_compressed(tmp_path, lats=lats, lons=lons, **grids)
            os.rename(tmp_path, path)
            logger.debug('Wrote %s' % path)

        self._converted = True

        missing = set(self.prefixes) - set(raw.keys())
        if missing:
            logger.warning(
                'Slabs missing in Slab2 archive: %s'
                % ', '.join(sorted(missing)))

    def get_slab(self, prefix):
        '''
        Get geometry of a slab.

        :param prefix: three-letter abbreviation of the slab, see
            :py:func:`get_slab_prefixes`
        :returns: :py:class:`Slab` object
        '''

        if prefix not in prefix_to_slab_name:
            raise Slab2Error(
                'Unknown slab: "%s". Available: %s' % (
                    prefix, ', '.join(self.prefixes)))

        if prefix not in self._slabs:
            path = self._cache_path(prefix)
            if not op.exists(path) and not self._converted:
                self.convert()

            if not op.exists(path):
                raise Slab2Error('No data available for slab "%s".' % prefix)

            with num.load(path) as data:
                grids = dict(
                    (parameter, data[parameter])
                    for parameter in get_parameter_names()
                    if parameter in data)

                self._slabs[prefix] = Slab(
                    prefix, data['lats'], data['lons'], **grids)

        return self._slabs[prefix]

    def get_slabs(self, prefixes=None):
        '''
        Get geometries of several slabs.

        :param prefixes: list of three-letter abbreviations of the slabs to
            load, defaults to all slabs
        :returns: list of :py:class:`Slab` objects
        '''

        if prefixes is None:
            prefixes = self.prefixes

        return [self.get_slab(prefix) for prefix in prefixes]


__all__ = [
    'Slab2Error',
    'Slab',
    'Slab2',
    'get_slab_names',
    'get_slab_prefixes',
    'get_parameter_names',
]
