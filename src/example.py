import os
from pyrocko import util
import logging


logger = logging.getLogger('pyrocko.example')


class DownloadError(Exception):
    pass


def get_example_data(filename, url=None):
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
        util.download_file(url, os.path.join(os.get_cwd(), filename))

    return filename
