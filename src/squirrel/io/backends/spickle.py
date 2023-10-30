# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Squirrel IO adaptor to read YAML files.
'''

import logging

from pyrocko.io.io_common import get_stats, touch  # noqa
from pyrocko import guts
from pyrocko.io.io_common import FileLoadError

logger = logging.getLogger('psq.io.spickle')


def provided_formats():
    return ['spickle']


def detect(first512):
    return 'spickle' if first512.startswith(b'SPICKLE') else None


def iload(format, file_path, segment, content):
    from pyrocko.io import stationxml

    with open(file_path, 'rb') as f:
        for obj, segment in guts._iload_all_spickle_internal(
                f, offset=segment):

            if isinstance(obj, stationxml.FDSNStationXML):
                from .stationxml import iload_stationxml
                yield from iload_stationxml(obj, segment, content)
            else:
                raise FileLoadError(
                    'Stored object of type %s is not supported by Squirrel '
                    'framework. Found at offset %s in spickle file: %s'
                    % (str(type(obj)), segment, file_path))
