# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import os

from pyrocko.guts import Object, String
from pyrocko import config

from . import error

guts_prefix = 'pf'


g_db_filename = 'nuts.sqlite'
g_cache_dirname = 'cache'


def get_squirrel_path(path=None):
    if path is None:
        path = os.curdir

    path = os.path.abspath(path)
    if os.path.isdir(path) and os.path.exists(
            os.path.join(path, g_db_filename)):
        return path

    while True:
        for entry in ['squirrel', '.squirrel']:
            candidate = os.path.join(path, entry)
            if os.path.isdir(candidate) \
                    and os.path.exists(os.path.join(candidate, g_db_filename)):

                return candidate

        path_new = os.path.dirname(path)
        if path_new == path:
            break

        path = path_new

    return os.path.join(config.config().cache_dir, 'squirrel')


def get_environment(path=None):
    if path is None:
        path = os.curdir

    squirrel_path = get_squirrel_path(path)

    return SquirrelEnvironment.make(squirrel_path)


def init_environment(path=None):
    if path is None:
        path = os.curdir

    squirrel_path = os.path.abspath(os.path.join(path, '.squirrel'))
    try:
        os.mkdir(squirrel_path)
    except OSError:
        raise error.SquirrelError(
            'Cannot create squirrel directory: %s' % squirrel_path)

    from .base import Squirrel
    env = SquirrelEnvironment.make(squirrel_path)
    sq = Squirrel(env)
    del sq


class SquirrelEnvironment(Object):
    database_path = String.T(optional=True)
    cache_path = String.T(optional=True)
    persistent = String.T(optional=True)

    @classmethod
    def make(cls, squirrel_path):
        return cls(
            database_path=os.path.join(squirrel_path, g_db_filename),
            cache_path=os.path.join(squirrel_path, g_cache_dirname))


__all__ = [
    'get_squirrel_path',
    'get_environment',
    'init_environment',
    'SquirrelEnvironment']
