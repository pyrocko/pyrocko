import urllib
import os
from pyrocko import util
import logging
from pyrocko.gf.ws import rget


logger = logging.getLogger('pyrocko.tutorials.util')


class DownloadError(Exception):
    pass


def get_tutorial_data(filename, url=None):
    '''
    Download example data file needed in tutorials.

    The file is downloaded to given ``filename``. If there already exists a
    file with that name, nothing is done.

    :param filename: name of the required file
    :param url: if not ``None`` get file from given URL otherwise fetch it from
        http://data.pyrocko.org/examples/<filename>.
    :returns: ``filename``
    '''

    if not os.path.exists(filename):
        url = 'http://data.pyrocko.org/examples/' + filename
        logger.info('Download %s' % url)
        try:
            try:
                rget(url, filename)
            except urllib.error.HTTPError as e:
                logger.debug(e)
                util.download_file(url, filename)

            logger.info('Finished Download')

        except Exception:
            raise DownloadError('could not download file from %s to %s' % (
                url, filename))


    return filename
