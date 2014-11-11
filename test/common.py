import os
from pyrocko import util


def test_data_file(fn):
    fpath = os.path.join(os.path.split(__file__)[0], 'data', fn)
    if not os.path.exists(fpath):
        url = 'http://kinherd.org/pyrocko_test_data/' + fn
        util.download_file(url, fpath)

    return fpath
