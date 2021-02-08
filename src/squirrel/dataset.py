# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import os.path as op
import logging

from pyrocko.guts import List, load

from ..has_paths import HasPaths
from .client.base import Source
from .error import SquirrelError

guts_prefix = 'pf'

logger = logging.getLogger('pyrocko.squirrel.dataset')


class Dataset(HasPaths):
    '''
    Dataset description.
    '''
    sources = List.T(Source.T())

    def setup(self, squirrel, check=True, progress_viewer='terminal'):
        for source in self.sources:
            source.setup(
                squirrel, check=check, progress_viewer=progress_viewer)


def read_dataset(path):
    '''
    Read dataset description file.
    '''
    try:
        dataset = load(filename=path)
    except OSError:
        raise SquirrelError(
            'Cannot read dataset file: %s' % path)

    if not isinstance(dataset, Dataset):
        raise SquirrelError('Invalid dataset file "%s".' % path)

    dataset.set_basepath(op.dirname(path) or '.')
    return dataset
