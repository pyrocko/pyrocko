import logging
import tarfile
from os import path as op
from collections import OrderedDict

from glob import glob

from pyrocko import util, config, geometry
from pyrocko.model import Geometry

from tempfile import mkdtemp
import numpy as num

logger = logging.getLogger('pyrocko.dataset.slab2')

citation = '''
Hayes, G., 2018, Slab2 - A Comprehensive Subduction Zone Geometry Model:
U.S. Geological Survey data release, https://doi.org/10.5066/F7PV6JNV.
'''

prefix_to_slab_name = {
    'alu': 'Alaska',
    'cal': 'Calabria',
    'cam': 'Central America',
    'cas': 'Cascadia',
    'cot': 'Cotabo',
    'car': 'Caribbean',
    'hal': 'Halmahera',
    'hel': 'Helenic',
    'him': 'Himalaya',
    'hin': 'Hindu Kush',
    'izu': 'Izu-Bonin',
    'ker': 'Kermadec',
    'kur': 'Kamchatka-Kuril Islands Japan',
    'mak': 'Makran',
    'man': 'Manila Trench',
    'mue': 'Muertos Trough',
    'pam': 'Pamir',
    'phi': 'Philippines',
    'png': 'NewGuinea',
    'puy': 'Puysegur',
    'ryu': 'Ryukyu',
    'sam': 'South America',
    'sco': 'Scotia Sea',
    'sol': 'Solomon Islands',
    'sul': 'Sulawesi',
    'sum': 'Sumatra-Java',
    'van': 'Vanuatu',
}

abbr_to_parameter = OrderedDict(
    dep='Depth',
    unc='Depth Uncertainty',
    str='Strike',
    dip='Dip',
    thk='Thickness',
)


def get_slab_names():
    return list(prefix_to_slab_name.values())


def get_slab_prefixes():
    return list(prefix_to_slab_name.keys())


class Slab2(object):

    SLAB2_URL = 'https://www.sciencebase.gov/catalog/file/get/5aa1b00ee4b0b1c392e86467?f=__disk__d5%2F91%2F39%2Fd591399bf4f249ab49ffec8a366e5070fe96e0ba'  # noqa

    def __init__(self):

        self.homedir_slab2 = config.config().slab2_dir
        self.fname_slab2 = op.join(self.homedir_slab2, 'slab2data.tar.gz')
        self.tmp_xyz_slab2 =  '/tmp/slab2_raw_data' # mkdtemp(prefix='slab2_raw_data')
        self.slab_prefixes = get_slab_prefixes()
        self.slab_names = get_slab_names()
        self.slab_parameters = list(abbr_to_parameter.keys())
        self.dtype = 'float32'
        self.dlon = self.dlat = 0.05   # [deg]

        if not op.exists(self.fname_slab2):
            self.download()

        self.slabs = []

    def slab_filename(self, slabname):
        return slabname + '.bin'

    def slab_geometry_path(self, slabname):
        return op.join(self.homedir_slab2, self.slab_filename(slabname))

    def slab_raw_data_path(self, raw_slab_path, param):
        return glob('{}*{}*'.format(raw_slab_path, param))[0]

    @property
    def n_parameters(self):
        return len(self.slab_parameters) + 2   # location

    def download(self):
        logger.info('Downloading Slab2 geometry data...')
        util.ensuredirs(self.fname_slab2)
        util.download_file(self.SLAB2_URL, self.fname_slab2)

    def unpack(self):

        def xyz_files(members):
            for tarinfo in members:
                if op.splitext(tarinfo.name)[1] == ".xyz":
                    logger.debug('Unpacking %s ...' % op.basename(tarinfo.name))
                    yield tarinfo

        if not op.exists(self.fname_slab2):
            raise ValueError(
                'The raw file does not exist at %s. Please download first!'
                % self.fname_slab2)

        logger.info('Unpacking Slab2 files ...')
        tar = tarfile.open(self.fname_slab2, mode='r|gz')
        tar.extractall(path=self.tmp_xyz_slab2, members=xyz_files(tar))
        tar.close()

    def convert_slab_geometry_to_bin(self, slabprefix, denan=False):

        if not op.exists(self.tmp_xyz_slab2):
            self.unpack()

        slab_bin_data_path = self.slab_geometry_path(slabprefix)
        raw_slab_path = op.join(
            self.tmp_xyz_slab2, 'Slab2Distribute_Mar2018/Slab2_TXT', slabprefix)
        logger.debug('Looking for raw data with %s' % raw_slab_path)

        file_path = self.slab_raw_data_path(raw_slab_path, 'dep')
        logger.debug('Assessing raw data at %s' % file_path)
        lonlatdepth = num.loadtxt(file_path, delimiter=',', dtype=self.dtype)

        npoints = lonlatdepth.shape[0]
        slab_data = num.empty((npoints, self.n_parameters), dtype=self.dtype)
        print(slab_data.shape)
        print(lonlatdepth)
        slab_data[:, 0:3] = lonlatdepth

        for i, param in enumerate(self.slab_parameters[1::]):
            file_path = self.slab_raw_data_path(raw_slab_path, param)
            logger.debug('Assessing raw data at %s' % file_path)
            tmp = num.loadtxt(file_path, delimiter=',', dtype=self.dtype)
            slab_data[:, i + 3] = tmp[:, 2]    # just extract parameter

        print('data shape', slab_data.shape)
        # denan = slab_data[~num.isnan(slab_data).any(axis=1)]
        if denan:
            logger.info('Removing Nans from data...')
            slab_data = slab_data[~num.isnan(slab_data).any(axis=1)]

        print('data shape before writing', slab_data.shape)
        with open(slab_bin_data_path, 'w') as f:
            if slab_data is not None:
                slab_data.tofile(f)

    def convert_all(self):
        logger.debug('Converting slab raw data to pyrocko format...')
        for slabprefix in self.slab_prefixes:
            self.convert_slab_geometry_to_bin(slabprefix)

    def get_slab_data(self, slabprefix, force_rewrite=False):

        slab_bin_data_path = self.slab_geometry_path(slabprefix)

        if not op.exists(slab_bin_data_path) or force_rewrite:
            logger.debug(
                'Slab2 geometry for %s does not exist, unpacking ...'
                '!' % slab_bin_data_path)
            self.convert_slab_geometry_to_bin(slabprefix)

        logger.debug(
            'Slab2 geometry for %s exists, loading ...!' % slabprefix)
        with open(slab_bin_data_path, 'r') as f:
            data = num.fromfile(f, dtype=self.dtype)

        return data.reshape((-1, self.n_parameters))

    def get_slab_geometry(self, slabprefix):
        from pyrocko.cake import earthradius
        from pyrocko.model import Geometry, Event

        data = self.get_slab_data(slabprefix)
        print(data.shape)
        lonmin, latmin = data[:, 0:2].min(0)
        lonmax, latmax = data[:, 0:2].max(0)
   #
        nlat = num.round((latmax - latmin) / self.dlat)
        nlon = num.round((lonmax - lonmin) / self.dlon)
   #     print(latmin, latmax, lonmin, lonmax, nlat, nlon)
   #
   #     #vlats = num.arange(latmin, latmax + self.dlat, self.dlat)
   #     vlats = latmin + num.arange(nlat + 1) * self.dlat
   #     vlons = lonmin + num.arange(nlon + 1) * self.dlon
   #
   #     lons, lats = num.meshgrid(vlons, vlats)
   #     print(lats[0:5, 0:5])
   #     print(lons[0:5, 0:5])
   #     #vlons = num.arange(lonmin, lonmax + self.dlon, self.dlon)
   #     print('mesh', vlats.min(), vlats.max(), vlons.min(), vlons.max())
   #
        depth = data[:, 2]

        # leave unnormalised as geometry does erthradius rescaling
        vertices = num.zeros((data.shape[0], 5))
        vertices[:, 0] = data[:, 1]
        vertices[:, 1] = data[:, 0]
        vertices[:, 4] = depth

        faces = geometry.topo_faces_quad(nlat, nlon)
        nanmask = num.isnan(depth[faces])

        print(vertices.shape, faces.shape)

        vertices = vertices[~num.isnan(vertices).any(axis=1)]
        print('verts', vertices)
        centers = geometry.face_centers(vertices, faces)
        print(centers.shape)
        print(centers[0:5,:])
        event = Event(lat=vlats.mean(), lon=vlons.mean(), depth=depth.mean())

  #      geom = Geometry(times=num.zeros(1), event=event)
  #      geom.setup(vertices, faces)
        #geom.add_property((('slip', 'float64', sub_headers)), srf_slips)

