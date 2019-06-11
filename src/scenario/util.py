# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division, print_function

from pyrocko import gf
from .error import CannotCreatePath


def remake_dir(dpath, force):
    try:
        return gf.store.remake_dir(dpath, force)

    except gf.CannotCreate as e:
        raise CannotCreatePath(str(e))
