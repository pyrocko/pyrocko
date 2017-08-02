import os
from pyrocko import util


def test_data_file_no_download(fn):
    return os.path.join(os.path.split(__file__)[0], fn)


def get_tutorial_data(filename):
    '''Download data needed in tutorials.

    Data is hosted at kinherd.org

    :param filename: Name of the required file
    '''
    fpath = test_data_file_no_download(filename)
    if not os.path.exists(fpath):
        url = 'http://kinherd.org/pyrocko_tutorial_data/' + filename
        print('Download %s' % url)
        util.download_file(url, fpath)

    return fpath
