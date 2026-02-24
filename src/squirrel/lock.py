# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Directory locking utility.
'''

import os
import logging

op = os.path
logger = logging.getLogger('psq.lock')


class LockDir(object):

    def __init__(self, path):
        self._path = path
        self._lockfile = op.join(path, '.buried')

    def __enter__(self):
        try:
            with open(self._lockfile, 'xb') as f:
                f.write(b'')
        except FileExistsError:
            raise EnvironmentError('Directory "%s" is locked' % self._path)

        logger.debug('Locked directory "%s"', self._path)
        return self

    def __exit__(self, type, value, traceback):
        try:
            os.remove(self._lockfile)
            logger.debug('Unlocked directory "%s"', self._path)
        except FileNotFoundError:
            logger.warning(
                'Lockfile "%s" was removed unintentionally', self._lockfile)
