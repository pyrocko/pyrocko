from __future__ import division, print_function, absolute_import
from pyrocko import trace, pile, io, config, util

import unittest
import numpy as num
import tempfile
import random
import os
from random import choice as rc
from os.path import join as pjoin


def numeq(a, b, eps):
    return num.all(num.abs(num.array(a) - num.array(b)) < eps)


def makeManyFiles(nfiles, nsamples, networks, stations, channels, tmin):

    datadir = tempfile.mkdtemp()
    traces = []
    deltat = 1.0
    for i in range(nfiles):
        ctmin = tmin+i*nsamples*deltat  # random.randint(1,int(time.time()))

        data = num.ones(nsamples)
        traces.append(
            trace.Trace(
                rc(networks), rc(stations), '', rc(channels),
                ctmin, None, deltat, data))

    fnt = pjoin(
        datadir,
        '%(network)s-%(station)s-%(location)s-%(channel)s-%(tmin)s.mseed')

    io.save(traces, fnt, format='mseed')

    return datadir


class PileTestCase(unittest.TestCase):

    def testPileTraversal(self):
        import shutil
        config.show_progress = False
        nfiles = 200
        nsamples = 1000

        abc = 'abcdefghijklmnopqrstuvwxyz'

        def rn(n):
            return ''.join([random.choice(abc) for i in range(n)])

        stations = [rn(4) for i in range(10)]
        channels = [rn(3) for i in range(3)]
        networks = ['xx']

        tmin = 1234567890
        datadir = makeManyFiles(
            nfiles, nsamples, networks, stations, channels, tmin)
        filenames = util.select_files([datadir], show_progress=False)
        cachedir = pjoin(datadir, '_cache_')
        p = pile.Pile()
        p.load_files(filenames=filenames, cache=pile.get_cache(cachedir),
                     show_progress=False)

        assert set(p.networks) == set(networks)
        assert set(p.stations) == set(stations)
        assert set(p.channels) == set(channels)

        ntr = 0
        for tr in p.iter_traces():
            ntr += 1
            assert tr.data_len() == nsamples

        assert ntr == nfiles

        toff = 0
        while toff < nfiles*nsamples:

            trs, loaded1 = p.chop(tmin+10, tmin+200)
            for tr in trs:
                assert num.all(tr.get_ydata() == num.ones(190))

            trs, loaded2 = p.chop(tmin-100, tmin+100)
            for tr in trs:
                assert len(tr.get_ydata()) == 100

            loaded = loaded1 | loaded2
            while loaded:
                file = loaded.pop()
                file.drop_data()

            toff += nsamples

        s = 0
        for traces in p.chopper(tmin=None, tmax=p.tmax+1., tinc=122.,
                                degap=False):
            for tr in traces:
                s += num.sum(tr.ydata)

        assert int(round(s)) == nfiles*nsamples

        for fn in filenames:
            os.utime(fn, None)

        p.reload_modified()

        pile.get_cache(cachedir).clean()
        shutil.rmtree(datadir)

    def testMemTracesFile(self):
        tr = trace.Trace(ydata=num.arange(100, dtype=num.float))

        f = pile.MemTracesFile(None, [tr])
        p = pile.Pile()
        p.add_file(f)
        for tr in p.iter_all(include_last=True):
            assert numeq(tr.ydata, num.arange(100, dtype=num.float), 0.001)


if __name__ == "__main__":
    util.setup_logging('test_pile', 'warning')
    unittest.main()
