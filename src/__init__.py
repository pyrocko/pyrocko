# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

try:
    from .info import *  # noqa
    __version__ = version  # noqa
except ImportError:
    pass  # not available in dev mode

grumpy = 0  # noqa


def get_logger():
    from .util import logging
    return logging.getLogger('pyrocko')


class ExternalProgramMissing(Exception):
    pass


def make_squirrel(*args, **kwargs):
    from pyrocko.squirrel import Squirrel
    return Squirrel(*args, **kwargs)


def sparrow(*args, **kwargs):
    '''
    Start Sparrow.

    Calls :py:func:`pyrocko.gui.sparrow.main`.
    '''

    from pyrocko import deps
    deps.require('vtk')

    from pyrocko.gui.sparrow.main import main
    return main(*args, **kwargs)


def drum(*args, **kwargs):
    '''
    Start Drum Plot.

    Calls :py:func:`pyrocko.gui.drum.main`.
    '''

    from pyrocko.gui.drum.main import main
    return main(*args, **kwargs)
