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


class Local(base.Source, has_paths.HasPaths):
    paths = List.T(has_paths.Path.T())
    kinds = List.T(ContentKind.T(), optional=True)
    format = FileFormat.T(default='detect')

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
    'Local']
