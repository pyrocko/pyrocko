# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Manage separate databases and caches for different user projects.

Squirrel based applications can either use the user's global database which
lives in Pyrocko's global cache directory (by default under
``'$HOME/.pyrocko/cache/squirrel'``) or a project specific local database which
can be conveniently created in the top level directory of a user's project
under ``'.squirrel'`` or ``'squirrel'``. The choice of database and associated
cache directory locations is referred to here as the Squirrel environment. This
module provides functions to create local environments and to look for a usable
environment in the hierarchy of a user's project directory.
'''

import os
import logging

from pyrocko.guts import String, load
from pyrocko import config
from pyrocko.has_paths import HasPaths, Path

from . import error
from .database import close_database

logger = logging.getLogger('psq.environment')

guts_prefix = 'squirrel'


g_db_filename = 'nuts.sqlite'
g_cache_dirname = 'cache'
g_config_filename = 'config.yaml'


def get_config_path(squirrel_path):
    return os.path.join(squirrel_path, g_config_filename)


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
    '''
    Get default Squirrel environment relevant for a given file system path.

    :param path:
        Directory path to use as starting point for detection of the Squirrel
        environment. By default, the current working directory is used as
        starting point. When searching for a usable environment the directory
        ``'.squirrel'`` or ``'squirrel'`` in the current (or starting point)
        directory is used if it exists, otherwise the parent directories are
        search upwards for the existence of such a directory. If no such
        directory is found, the user's global Squirrel environment, usually
        ``'$HOME/.pyrocko/cache/squirrel'``, is used.

    :returns:
        :py:class:`Environment` object containing the detected database
        and cache directory paths.
    '''
    if path is None:
        path = os.curdir

    squirrel_path = get_squirrel_path(path)

    if os.path.exists(get_config_path(squirrel_path)):
        return Environment.load(squirrel_path)
    else:
        return Environment.make(squirrel_path)


def init_environment(path=None):
    '''
    Initialize empty Squirrel environment.

    :param path:
        Path to the directory where the new environment's ``'.squirrel'``
        directory should be created. If set to ``None``, the current working
        directory is used.

    If a ``'.squirrel'`` directory already exists at the given location,
    :py:exc:`~pyrocko.squirrel.error.SquirrelError` is raised.
    '''
    if path is None:
        path = os.curdir

    squirrel_path = os.path.join(path, '.squirrel')
    try:
        logger.info(
            'Creating squirrel environment directory: %s'
            % os.path.abspath(squirrel_path))
        os.mkdir(squirrel_path)
    except OSError:
        raise error.SquirrelError(
            'Cannot create squirrel directory: %s' % squirrel_path)

    from .base import Squirrel
    env = Environment.make(squirrel_path)
    env.dump(filename=get_config_path(squirrel_path))
    sq = Squirrel(env)
    database = sq.get_database()
    del sq
    close_database(database)


class Environment(HasPaths):
    '''
    Configuration object providing paths to database and cache.
    '''

    database_path = Path.T(optional=True)
    cache_path = Path.T(optional=True)
    persistent = String.T(optional=True)

    @classmethod
    def make(cls, squirrel_path):
        env = cls(
            database_path=g_db_filename,
            cache_path=g_cache_dirname)

        env.set_basepath(squirrel_path)
        return env

    @classmethod
    def load(cls, squirrel_path):
        path = get_config_path(squirrel_path)
        try:
            env = load(filename=path)
        except OSError:
            raise error.SquirrelError(
                'Cannot read environment config file: %s' % path)

        if not isinstance(env, Environment):
            raise error.SquirrelError(
                'Invalid environment config file "%s".' % path)

        env.set_basepath(squirrel_path)
        return env


__all__ = [
    'get_squirrel_path',
    'get_environment',
    'init_environment',
    'Environment']
