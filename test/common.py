import os
from pyrocko import util


def test_data_file_no_download(fn):
    return os.path.join(os.path.split(__file__)[0], 'data', fn)


def test_data_file(fn):
    fpath = test_data_file_no_download(fn)
    if not os.path.exists(fpath):
        url = 'http://kinherd.org/pyrocko_test_data/' + fn
        util.download_file(url, fpath)

    return fpath
