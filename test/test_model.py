from __future__ import division, print_function, absolute_import
import unittest
import math
import tempfile
import shutil

from pyrocko import model, util, trace, orthodrome, guts, moment_tensor
import numpy as num
from os.path import join as pjoin

d2r = num.pi/180.

eps = 1e-15


def assertOrtho(a, b, c):
    xeps = max(abs(max(a)), abs(max(b)), abs(max(c)))*eps
    assert abs(num.dot(a, b)) < xeps
    assert abs(num.dot(a, c)) < xeps
    assert abs(num.dot(b, c)) < xeps


def near(a, b, eps):
    return abs(a-b) < eps


class ModelTestCase(unittest.TestCase):

    def testIOEventOld(self):
        tempdir = tempfile.mkdtemp()
        fn = pjoin(tempdir, 'event.txt')
        e1 = model.Event(
            10., 20., 1234567890., 'bubu', region='taka tuka land',
            magnitude=5.1, magnitude_type='Mw')
        e1.olddump(fn)
        e2 = model.Event(load=fn)
        assert e1.region == e2.region
        assert e1.name == e2.name
        assert e1.lat == e2.lat
        assert e1.lon == e2.lon
        assert e1.time == e2.time
        assert e1.region == e2.region
        assert e1.magnitude == e2.magnitude
        assert e1.magnitude_type == e2.magnitude_type
        shutil.rmtree(tempdir)

    def testIOEvent(self):
        tempdir = tempfile.mkdtemp()
        fn = pjoin(tempdir, 'event.txt')
        e1 = model.Event(
            10., 20., 1234567890., 'bubu', region='taka tuka land',
            moment_tensor=moment_tensor.MomentTensor(strike=45., dip=90),
            magnitude=5.1, magnitude_type='Mw')
        guts.dump(e1, filename=fn)
        e2 = guts.load(filename=fn)
        assert e1.region == e2.region
        assert e1.name == e2.name
        assert e1.lat == e2.lat
        assert e1.lon == e2.lon
        assert e1.time == e2.time
        assert e1.region == e2.region
        assert e1.magnitude == e2.magnitude
        assert e1.magnitude_type == e2.magnitude_type
        shutil.rmtree(tempdir)

    def testMissingComponents(self):

        ne = model.Channel('NE', azimuth=45., dip=0.)
        se = model.Channel('SE', azimuth=135., dip=0.)

        station = model.Station('', 'STA', '', 0., 0., 0., channels=[ne, se])

        mat = station.projection_to_enu(('NE', 'SE', 'Z'), ('E', 'N', 'U'))[0]
        assertOrtho(mat[:, 0], mat[:, 1], mat[:, 2])

        n = model.Channel('D', azimuth=0., dip=90.)
        station.set_channels([n])
        mat = station.projection_to_enu(('N', 'E', 'D'), ('E', 'N', 'U'))[0]
        assertOrtho(mat[:, 0], mat[:, 1], mat[:, 2])

    def testIOStations(self):
        tempdir = tempfile.mkdtemp()
        fn = pjoin(tempdir, 'stations.txt')

        ne = model.Channel('NE', azimuth=45., dip=0.)
        se = model.Channel('SE', azimuth=135., dip=0.)
        stations = [
            model.Station('', sta, '', 0., 0., 0., channels=[ne, se])
            for sta in ['STA1', 'STA2']]

        model.dump_stations(stations, fn)
        stations = model.load_stations(fn)

        shutil.rmtree(tempdir)

    def testProjections(self):
        km = 1000.

        ev = model.Event(lat=-10, lon=150., depth=0.0)

        for azi in num.linspace(0., 360., 37):
            lat, lon = orthodrome.ne_to_latlon(
                ev.lat, ev.lon, 10.*km * math.cos(azi), 10.*km * math.sin(azi))

            sta = model.Station(lat=lat, lon=lon)
            sta.set_event_relative_data(ev)
            sta.set_channels_by_name('BHZ', 'BHE', 'BHN')

            r = 1.
            t = 1.
            traces = [
                trace.Trace(
                    channel='BHE',
                    ydata=num.array([math.sin(azi)*r+math.cos(azi)*t])),
                trace.Trace(
                    channel='BHN',
                    ydata=num.array([math.cos(azi)*r-math.sin(azi)*t])),
            ]

            for m, in_channels, out_channels in sta.guess_projections_to_rtu():
                projected = trace.project(traces, m, in_channels, out_channels)

            def g(traces, cha):
                for tr in traces:
                    if tr.channel == cha:
                        return tr

            r = g(projected, 'R')
            t = g(projected, 'T')
            assert(near(r.ydata[0], 1.0, 0.001))
            assert(near(t.ydata[0], 1.0, 0.001))


if __name__ == "__main__":
    util.setup_logging('test_trace', 'warning')
    unittest.main()
