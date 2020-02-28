import logging
import tarfile
from os import path as op
from collections import OrderedDict

from glob import glob

from pyrocko import util, config
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

class Slab2(object):

    SLAB2_URL = 'https://www.sciencebase.gov/catalog/file/get/5aa1b00ee4b0b1c392e86467?f=__disk__d5%2F91%2F39%2Fd591399bf4f249ab49ffec8a366e5070fe96e0ba'  # noqa

    def __init__(self):

        self.homedir_slab2 = config.config().slab2_dir
        self.fname_slab2 = op.join(self.homedir_slab2, 'slab2data.tar.gz')
        self.tmp_xyz_slab2 = mkdtemp(prefix='slab2_raw_data')
        self.slab_prefixes = list(prefix_to_slab_name.keys())
        self.slab_parameters = list(abbr_to_parameter.kexs())

        if not op.exists(self.fname_slab2):
            self.download()

        self.slabs = []

    def n_parameters(self):
        return len(self.slab_parameters) + 2   # location

    def download(self):
        logger.info('Downloading Slab2 geometry data...')
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

        tar = tarfile.open(self.fname_slab2, mode='r|gz')
        tar.extractall(path=self.tmp_xyz_slab2, members=xyz_files(tar))
        tar.close()

    def convert_to_bin(self):

        for slab in self.slab_prefixes:
            slab_path = op.join(self.tmp_xyz_slab2, slab)
            location = glob('{]*{}*'.format(slab_path, 'dep'))
            slab_files = glob('{]*'.format(slab_path))
            
            latlondepth = num.loadtxt(location, delimiter=',', dtype='float64')
            npoints = latlondepth.shape[0]
            slab_data = num.empty((npoints, self.n_parameters, dtype='float64'))
            slab_data[:, 0:2] = latlondepth
            for i, param in enumerate(self.slab_parameters):
                tmp = num.loadtxt(param, delimiter=',', dtype='float64')
                slab_data[:, i + 2] = tmp[:, 2]    # just extract parameter

