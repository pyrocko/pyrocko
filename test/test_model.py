from __future__ import division, print_function, absolute_import
import unittest
import math
import tempfile
import shutil

from pyrocko import model, util, trace, orthodrome, guts, moment_tensor
from pyrocko.guts import load
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
        tempdir = tempfile.mkdtemp(prefix='pyrocko-model')
        fn = pjoin(tempdir, 'event.txt')
        e1 = model.Event(
            10., 20., 1234567890., 'bubu', region='taka tuka land',
            magnitude=5.1, magnitude_type='Mw',
            tags=['cluster:-1', 'custom_magnitude:2.5'])

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
        assert e1.tags == e2.tags
        shutil.rmtree(tempdir)

    def testIOEvent(self):
        tempdir = tempfile.mkdtemp(prefix='pyrocko-model')
        fn = pjoin(tempdir, 'event.txt')
        e1 = model.Event(
            10., 20., 1234567890., 'bubu', region='taka tuka land',
            moment_tensor=moment_tensor.MomentTensor(strike=45., dip=90),
            magnitude=5.1, magnitude_type='Mw',
            tags=['cluster:-1', 'custom_magnitude:2.5'])

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
        assert e1.get_hash() == e2.get_hash()
        assert e1.tags == e2.tags

        fn2 = pjoin(tempdir, 'events.txt')
        guts.dump_all([e1, e2], filename=fn2)

        with self.assertRaises(model.OneEventRequired):
            model.load_one_event(fn2)

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
        tempdir = tempfile.mkdtemp(prefix='pyrocko-model')
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

    def testProjectionsZOnly(self):
        km = 1000.

        ev = model.Event(lat=-10, lon=150., depth=0.0)

        for azi in num.linspace(0., 360., 37):
            lat, lon = orthodrome.ne_to_latlon(
                ev.lat, ev.lon, 10.*km * math.cos(azi), 10.*km * math.sin(azi))

            sta = model.Station(lat=lat, lon=lon)
            sta.set_event_relative_data(ev)
            sta.set_channels_by_name('BHZ', 'BHN', 'BHE')
            traces = [
                trace.Trace(
                    channel='BHZ',
                    ydata=num.array([1.0])),
            ]

            projected = []
            for m, in_channels, out_channels in sta.guess_projections_to_enu():
                projected.extend(trace.project(
                    traces, m, in_channels, out_channels))

            def g(traces, cha):
                for tr in traces:
                    if tr.channel == cha:
                        return tr

            z = g(projected, 'U')
            assert(near(z.ydata[0], 1.0, 0.001))

    def testGNSSCampaign(self):
        tempdir = tempfile.mkdtemp(prefix='pyrocko-model')
        fn = pjoin(tempdir, 'gnss_campaign.yml')

        nstations = 25

        lats = num.random.uniform(90, -90, nstations)
        lons = num.random.uniform(90, -90, nstations)

        shifts = num.random.uniform(-2.5, 2.5, (nstations, 3))
        sigma = num.random.uniform(-0.5, 0.5, (nstations, 3))

        correlations = num.random.uniform(-0.5, 0.5, (nstations, 3))

        campaign = model.gnss.GNSSCampaign()

        for ista in range(nstations):

            north = model.gnss.GNSSComponent(
                shift=float(shifts[ista, 0]),
                sigma=float(sigma[ista, 0]))
            east = model.gnss.GNSSComponent(
                shift=float(shifts[ista, 1]),
                sigma=float(sigma[ista, 1]))
            up = model.gnss.GNSSComponent(
                shift=float(shifts[ista, 2]),
                sigma=float(sigma[ista, 2]))

            station = model.gnss.GNSSStation(
                lat=float(lats[ista]),
                lon=float(lons[ista]),
                correlation_eu=float(correlations[ista, 0]),
                correlation_ne=float(correlations[ista, 1]),
                correlation_nu=float(correlations[ista, 2]),
                north=north,
                east=east,
                up=up)

            campaign.add_station(station)

        for sta in campaign.stations:
            sta.get_correlation_matrix(full=False)
            corr_arr = sta.get_correlation_matrix(full=True)
            num.testing.assert_array_equal(corr_arr, corr_arr.T)

        corr_arr = campaign.get_correlation_matrix()
        num.testing.assert_array_equal(corr_arr, corr_arr.T)

        corr_arr_ref = num.zeros(corr_arr.shape)

        for ista, sta in enumerate(campaign.stations):
            for ic, comp in enumerate([sta.north, sta.east, sta.up]):
                corr_arr_ref[ista*3+ic, ista*3+ic] = comp.sigma

            corr_arr_ref[ista*3, ista*3+1] = sta.correlation_ne
            corr_arr_ref[ista*3+1, ista*3] = sta.correlation_ne

            corr_arr_ref[ista*3, ista*3+2] = sta.correlation_nu
            corr_arr_ref[ista*3+2, ista*3] = sta.correlation_nu

            corr_arr_ref[ista*3+1, ista*3+2] = sta.correlation_eu
            corr_arr_ref[ista*3+2, ista*3+1] = sta.correlation_eu

        covar_arr = campaign.get_covariance_matrix()
        num.testing.assert_array_equal(covar_arr, covar_arr.T)

        covar_ref = num.zeros(covar_arr.shape)

        for ista, sta in enumerate(campaign.stations):
            for ic, comp in enumerate([sta.north, sta.east, sta.up]):
                covar_ref[ista*3+ic, ista*3+ic] = comp.sigma**2

            covar_ref[ista*3, ista*3+1] =\
                sta.correlation_ne * sta.north.sigma * sta.east.sigma
            covar_ref[ista*3+1, ista*3] =\
                sta.correlation_ne * sta.north.sigma * sta.east.sigma

            covar_ref[ista*3, ista*3+2] =\
                sta.correlation_nu * sta.north.sigma * sta.up.sigma
            covar_ref[ista*3+2, ista*3] =\
                sta.correlation_nu * sta.north.sigma * sta.up.sigma

            covar_ref[ista*3+1, ista*3+2] =\
                sta.correlation_eu * sta.east.sigma * sta.up.sigma
            covar_ref[ista*3+2, ista*3+1] =\
                sta.correlation_eu * sta.east.sigma * sta.up.sigma

        num.testing.assert_array_equal(covar_arr, covar_ref)

        campaign.dump(filename=fn)

        campaign2 = load(filename=fn)

        s1 = campaign.stations[0]

        s_add = s1.north + s1.north
        assert s_add.shift == (s1.north.shift + s1.north.shift)

        assert len(campaign.stations) == len(campaign2.stations)
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    util.setup_logging('test_trace', 'warning')
    unittest.main()
