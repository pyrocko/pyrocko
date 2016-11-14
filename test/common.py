import os
import time
from pyrocko import util


def test_data_file_no_download(fn):
    return os.path.join(os.path.split(__file__)[0], 'data', fn)


def test_data_file(fn):
    fpath = test_data_file_no_download(fn)
    if not os.path.exists(fpath):
        url = 'http://kinherd.org/pyrocko_test_data/' + fn
        util.download_file(url, fpath)

    return fpath


def benchmark(func):
    def stopwatch(*args):
        t0 = time.time()
        result = func(*args)
        elapsed = time.time() - t0
        util.logger.info('%s executed in %.8f' % (func.__name__, elapsed))
        return result
    return stopwatch
