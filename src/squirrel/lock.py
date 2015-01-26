# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function


class LockDir(object):

    def __init__(self, path):
        self._path = path

    def __enter__(self):
        # TODO: implement locking
        print('Would lock directory "%s" (todo)' % self._path)
        return self

    def __exit__(self, type, value, traceback):
        # TODO: implement unlocking
        print('Would unlock directory "%s" (todo)' % self._path)
