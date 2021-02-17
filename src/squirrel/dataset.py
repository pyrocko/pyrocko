# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import os.path as op
import logging

from pyrocko.guts import List, load, StringPattern

from ..has_paths import HasPaths
from .client.base import Source
from .error import SquirrelError
from .selection import re_persistent_name

guts_prefix = 'squirrel'

logger = logging.getLogger('psq.dataset')


class PersistentID(StringPattern):
    pattern = re_persistent_name


class Dataset(HasPaths):
    '''
    Dataset description.
    '''
    sources = List.T(Source.T())
    persistent = PersistentID.T(optional=True)

    def setup(self, squirrel, check=True, progress_viewer='terminal'):
        for source in self.sources:
            source.setup(
                squirrel, check=check, progress_viewer=progress_viewer)

    def get_squirrel(
            self,
            update=False,
            check=True,
            how_to_update='Avoiding dataset rescan. '
                          'Enable updating to force refresh or delete the '
                          'persistent selection for a clean start.'):

        from pyrocko.squirrel import base
        squirrel = base.Squirrel(persistent=self.persistent)

        if self.persistent and not squirrel.is_new():
            if not update:
                logger.info(
                    'Using existing persistent selection: %s'
                    % self.persistent)
                logger.info(how_to_update)
                return squirrel

            else:
                logger.info(
                    'Updating existing persistent selection: %s'
                    % self.persistent)

        squirrel.add_dataset(self, check=check)
        return squirrel


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


def from_dataset(path, update=False, check=True):
    ds = read_dataset(path)
    return ds.get_squirrel(update=update, check=check)


__all__ = [
    'PersistentID',
    'Dataset',
    'read_dataset',
    'from_dataset'
]
