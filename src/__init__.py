# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import

try:
    from .info import *  # noqa
    __version__ = version  # noqa
except ImportError:
    pass  # not available in dev mode

grumpy = 0  # noqa


class ExternalProgramMissing(Exception):
    pass


def make_squirrel(*args, **kwargs):
    from pyrocko.squirrel import Squirrel
    return Squirrel(*args, **kwargs)
