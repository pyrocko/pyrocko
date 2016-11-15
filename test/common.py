import os
import time
from pyrocko import util

benchmark_results = []


def test_data_file(fn):
    fpath = os.path.join(os.path.split(__file__)[0], 'data', fn)
    if not os.path.exists(fpath):
        url = 'http://kinherd.org/pyrocko_test_data/' + fn
        util.download_file(url, fpath)

    return fpath


class Benchmark(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.results = []

    def __call__(self, func):
        def stopwatch(*args):
            t0 = time.time()
            name = self.prefix + func.__name__
            result = func(*args)
            elapsed = time.time() - t0
            self.results.append((name, elapsed))
            return result
        return stopwatch

    def __str__(self):
        rstr = ['Benchmark results']
        indent = max([len(name) for name, _ in self.results])
        rstr.append('=' * (indent + 17))
        for res in self.results:
            rstr.append(
                '{0:<{indent}}{1:.8f} s'.format(*res, indent=indent+5))
        return '\n'.join(rstr)
