# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import logging

from pyrocko.guts import List, StringChoice
from pyrocko import has_paths
from . import base
from .. import io

guts_prefix = 'pf'

logger = logging.getLogger('pyrocko.squirrel.client.local')


class FileFormat(StringChoice):
    choices = ['detect'] + io.supported_formats()


class ContentKind(StringChoice):
    choices = io.base.g_content_kinds


class LocalData(base.Source, has_paths.HasPaths):
    '''
    A collection of local files attachable as a Squirrel data-source.
    '''
    paths = List.T(
        has_paths.Path.T(),
        help='Directory and file paths to add to the Squirrel '
             'instance. See :py:meth:`Squirrel.add() '
             '<pyrocko.squirrel.base.Squirrel.add>`.')
    kinds = List.T(
        ContentKind.T(),
        optional=True,
        help='Content kinds to be added to the Squirrel selection. By default '
             'all known content kinds are added.')
    format = FileFormat.T(
        default='detect',
        help='Assume files are of given format.')

    def setup(self, squirrel, check=True, progress_viewer='terminal'):
        squirrel.add(
            self.expand_path(self.paths),
            kinds=self.kinds,
            format=self.format,
            check=check,
            progress_viewer=progress_viewer)


__all__ = [
    'FileFormat',
    'ContentKind',
    'LocalData']
