from __future__ import division, print_function, absolute_import
import unittest
import platform
import os
import sys
import time
from pyrocko import util
import functools
import logging
import socket
import shutil
from io import StringIO

try:
    new_str = unicode
except NameError:
    new_str = str

import os.path as op
import tempfile

import numpy as num

from pyrocko import io, trace

logger = logging.getLogger('pyrocko.test.common')

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

benchmark_results = []

g_matplotlib_inited = False

H = 3600.
D = 3600.*24.


def get_gf_engine():
    from pyrocko import gf
    return gf.get_engine(
        [os.path.join(os.path.dirname(__file__), 'data', 'gf_stores')])


def have_gf_store(store_id):
    from pyrocko import gf
    engine = get_gf_engine()
    try:
        engine.get_store(store_id)
        return True, ''
    except gf.NoSuchStore:
        return False, 'GF store "%s" not available' % store_id


def matplotlib_use_agg():
    global g_matplotlib_inited
    if not g_matplotlib_inited:
        import matplotlib
        matplotlib.use('Agg')  # noqa
        g_matplotlib_inited = True


def test_data_file_no_download(fn):
    return os.path.join(os.path.split(__file__)[0], 'data', fn)


def make_dataset(dir=None, nstations=10, nchannels=3, tlen=10*D, deltat=0.01,
                 tinc=1*H):

    if dir is None:
        dir = tempfile.mkdtemp('_test_squirrel_dataset')

    tref = util.str_to_time('2015-01-01 00:00:00')

    nblocks = int(round(tlen / tinc))

    for istation in range(nstations):
        for ichannel in range(nchannels):
            for iblock in range(nblocks):
                tmin = tref + iblock*tinc
                nsamples = int(round(tinc/deltat))
                ydata = num.random.randint(-1000, 1001, nsamples).astype(
                    num.int32)

                tr = trace.Trace(
                    '', '%04i' % istation, '', '%03i' % ichannel,
                    tmin=tmin,
                    deltat=deltat,
                    ydata=ydata)

                io.save([tr], op.join(dir, '%s/%c/%b.mseed'))

    return dir


def test_data_file(fn):
    fpath = test_data_file_no_download(fn)
    if not os.path.exists(fpath):
        if not have_internet():
            raise unittest.SkipTest(
                'need internet access to download data file')

        url = 'http://data.pyrocko.org/testing/pyrocko/' + fn
        logger.info('downloading %s' % url)
        util.download_file(url, fpath)

    return fpath


def have_internet():
    try:
        return 0 < len([
            (s.connect(('8.8.8.8', 80)), s.close())
            for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]])

    except OSError:
        return False


require_internet = unittest.skipUnless(have_internet(), 'need internet access')


def skip_on(*exceptions):

    def skip_on(f):
        @functools.wraps(f)
        def wrap_f(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions as e:
                raise unittest.SkipTest('skipped due to %s: "%s"' % (
                    e.__class__.__name__, str(e)))

        return wrap_f

    return skip_on


def skip_on_download_error(f):

    @functools.wraps(f)
    def wrap_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except util.DownloadError:
            raise unittest.SkipTest('download failed')

    return wrap_f


def have_gui():
    is_windows = platform.system() == 'Windows'

    display = os.environ.get('DISPLAY', '')
    if not display and not is_windows:
        return False

    try:
        from pyrocko.gui.qt_compat import qc  # noqa
    except ImportError:
        return False

    return True


require_gui = unittest.skipUnless(have_gui(), 'no gui support configured')


def have_obspy():
    try:
        import obspy  # noqa
        return True
    except ImportError:
        return False


require_obspy = unittest.skipUnless(have_obspy(), 'obspy not installed')


class BenchmarkCM(object):
    def __init__(self, label, results):
        self._label = label
        self._results = results

    def __enter__(self):
        self._t0 = time.time()

    def __exit__(self, *args):
        self._results.append((self._label, time.time() - self._t0))


class Benchmark(object):
    def __init__(self, prefix=None):
        self.prefix = prefix or ''
        self.show_factor = False
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

    def labeled(self, label):
        def wrapper(func):
            @functools.wraps(func)
            def stopwatch(*args):
                t0 = time.time()
                result = func(*args)
                elapsed = time.time() - t0
                self.results.append((label, elapsed))
                return result
            return stopwatch
        return wrapper

    def run(self, label):
        return BenchmarkCM(label, self.results)

    def __str__(self, header=True):
        if not self.results:
            return 'No benchmarks ran'
        tmax = max([r[1] for r in self.results])

        rstr = ['Benchmark results']
        if self.prefix != '':
            rstr[-1] += ' - %s' % self.prefix

        if self.results:
            indent = max([len(name) for name, _ in self.results])
        else:
            indent = 0
        rstr.append('=' * (indent + 17))
        rstr.insert(0, rstr[-1])

        if not header:
            rstr = []

        for res in self.results:
            rstr.append(
                '{0:<{indent}}{1:.8f} s'.format(*res, indent=indent+5))
            if self.show_factor:
                if res[1] != 0.0:
                    rstr[-1] += '{0:8.2f} x'.format(tmax/res[1])
                else:
                    rstr[-1] += '???'

        if len(self.results) == 0:
            rstr.append('None ran!')

        return '\n\n' + '\n'.join(rstr) + '\n'

    def clear(self):
        self.results = []


class PyrockoExit(Exception):
    def __init__(self, res):
        Exception.__init__(self, str(res))
        self.result = res


class Capture(object):
    def __init__(self, tee=False):
        self.file = StringIO()
        self.tee = tee

    def __enter__(self):
        self.orig_stdout = sys.stdout
        self.orig_exit = sys.exit
        sys.stdout = self

        def my_exit(res):
            raise PyrockoExit(res)

        sys.exit = my_exit

    def __exit__(self, *args):
        sys.stdout = self.orig_stdout
        sys.exit = self.orig_exit

    def write(self, data):
        data = new_str(data)
        self.file.write(data)
        if self.tee:
            self.orig_stdout.write(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self.file.flush()

    def isatty(self):
        return False

    def getvalue(self):
        return self.file.getvalue()


def call(program, *args, **kwargs):
    if program == 'fomosto':
        from pyrocko.apps.fomosto import main
    else:
        assert False, 'program %s not available' % program

    # tee = True
    tee = kwargs.get('tee', False)
    logger.info('Calling: %s %s' % (program, ' '.join(args)))
    cap = Capture(tee=tee)
    with cap:
        main(list(args))

    return cap.getvalue()


def call_assert_usage(program, *args):
    res = None
    try:
        call(program, *args)
    except PyrockoExit as e:
        res = e.result

    assert res.startswith('Usage')


class chdir(object):

    def __init__(self, path):
        self._path = path

    def __enter__(self):
        self._oldwd = os.getcwd()
        os.chdir(self._path)

    def __exit__(self, *args):
        os.chdir(self._oldwd)


class run_in_temp(object):
    def __init__(self, path=None):
        self._must_delete = False
        self._path = path

    def __enter__(self):
        if self._path is None:
            from tempfile import mkdtemp
            self._path = mkdtemp(prefix='pyrocko-test')
            self._must_delete = True

        self._oldwd = os.getcwd()
        os.chdir(self._path)

    def __exit__(self, *args):
        os.chdir(self._oldwd)
        if self._must_delete:
            shutil.rmtree(self._path)
