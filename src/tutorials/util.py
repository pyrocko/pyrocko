import future
import urllib
import os
from pyrocko import util
import logging
from pyrocko.gf.ws import rget


logger = logging.getLogger('pyrocko.tutorials.util')


def get_tutorial_data(filename):
    '''Download data needed in tutorials.

    Data is hosted at kinherd.org

    :param filename: Name of the required file
    '''
    if not os.path.exists(filename):
        url = 'http://data.pyrocko.org/examples/' + filename
        print('Download %s' % url)
        try:
            rget(url, filename)
        except urllib.error.HTTPError as e:
            logger.debug(e)
            util.download_file(url, filename)
        print('Finished Download')

    return filename
