from __future__ import division, print_function, absolute_import

import unittest
import tempfile
import numpy as num
import logging
import os
from pyrocko import util, trace
from pyrocko.io import stationxml
from pyrocko.client import fdsn, iris

from .. import common

logger = logging.getLogger('pyrocko.test.test_fdsn')

stt = util.str_to_time


def fix_resp_units(fn):
    with open(fn, 'r') as fin:
        with open(fn + '.temp', 'w') as fout:
            for line in fin:
                line = line.replace('count - ', 'counts - ')
                fout.write(line)

    os.rename(fn + '.temp', fn)


class FDSNStationTestCase(unittest.TestCase):

    def test_read_samples(self):
        ok = False
        for fn in ['geeil.iris.xml', 'geeil.geofon.xml']:
            fpath = common.test_data_file(fn)
            x = stationxml.load_xml(filename=fpath)
            for network in x.network_list:
                assert network.code == 'GE'
                for station in network.station_list:
                    assert station.code == 'EIL'
                    for channel in station.channel_list:
                        assert channel.code[:2] == 'BH'
                        for stage in channel.response.stage_list:
                            ok = True

            assert ok

            pstations = x.get_pyrocko_stations()
            assert len(pstations) in (3, 4)
            for s in x.get_pyrocko_stations():
                assert len(s.get_channels()) == 3

            assert len(x.get_pyrocko_stations(
                time=stt('2010-01-15 10:00:00'))) == 1

            new = stationxml.FDSNStationXML.from_pyrocko_stations(pstations)
            assert len(new.get_pyrocko_stations()) in (3, 4)
            for s in new.get_pyrocko_stations():
                assert len(s.get_channels()) == 3

    @common.require_internet
    @common.skip_on_download_error
    def test_retrieve(self):
        for site in ['geofon', 'iris']:
            fsx = fdsn.station(site=site,
                               network='GE',
                               station='EIL',
                               level='channel')

            assert len(fsx.get_pyrocko_stations(
                time=stt('2010-01-15 10:00:00'))) == 1

    @common.require_internet
    @common.skip_on_download_error
    def test_dataselect(self):
        tmin = stt('2010-01-15 10:00:00')
        tmax = stt('2010-01-15 10:01:00')
        for site in ['geofon', 'iris']:
            fdsn.dataselect(site=site,
                            network='GE',
                            station='EIL',
                            starttime=tmin,
                            endtime=tmax)

    @common.require_internet
    @common.skip_on_download_error
    def test_dataselection(self):
        tmin = stt('2010-01-15 10:00:00')
        tmax = stt('2010-01-15 10:01:00')
        selection = [
            ('GE', 'EIL', '*', 'SHZ', tmin, tmax),
        ]

        fdsn.dataselect(site='geofon', selection=selection)
        fdsn.station(site='geofon', selection=selection, level='response')

    def test_read_big(self):
        for site in ['iris']:
            fpath = common.test_data_file('%s_1014-01-01_all.xml' % site)
            stationxml.load_xml(filename=fpath)

    # @unittest.skip('needs manual inspection')
    @common.require_internet
    @common.skip_on_download_error
    def test_response(self, ntest=4):
        tmin = stt('2014-01-01 00:00:00')
        tmax = stt('2014-01-02 00:00:00')
        sx = fdsn.station(
            site='iris',
            network='II',
            channel='?HZ',
            startbefore=tmin,
            endafter=tmax,
            level='channel', format='text', matchtimeseries=True)

        for nslc in sx.nslc_code_list[:ntest]:
            net, sta, loc, cha = nslc
            sxr = fdsn.station(
                site='iris',
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                startbefore=tmin,
                endafter=tmax,
                level='response', matchtimeseries=True)

            fi = iris.ws_resp(
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                tmin=tmin,
                tmax=tmax)

            _, fn = tempfile.mkstemp()
            fo = open(fn, 'wb')
            while True:
                d = fi.read(1024)
                if not d:
                    break

                fo.write(d)

            fo.close()

            fix_resp_units(fn)

            resp_sx = sxr.get_pyrocko_response(
                nslc, timespan=(tmin, tmax),
                fake_input_units='M/S')

            resp_er = trace.Evalresp(fn, target='vel', nslc_id=nslc, time=tmin)
            fmin = 0.001
            fmax = 100.

            for _, _, channel in sxr.iter_network_station_channels(
                    net, sta, loc, cha, timespan=(tmin, tmax)):
                if channel.response:
                    fmax = channel.sample_rate.value * 0.5

            f = num.exp(num.linspace(num.log(fmin), num.log(fmax), 500))

            t_sx = resp_sx.evaluate(f)
            t_er = resp_er.evaluate(f)
            import pylab as lab

            abs_dif = num.abs(num.abs(t_sx) - num.abs(t_er)) / num.max(
                num.abs(t_er))

            mda = num.mean(abs_dif[f < 0.4*fmax])

            pha_dif = num.abs(num.angle(t_sx) - num.angle(t_er))

            mdp = num.mean(pha_dif[f < 0.4*fmax])

            if mda > 0.05 or mdp > 0.04:

                lab.gcf().add_subplot(2, 1, 1)
                lab.plot(f, num.abs(t_sx), color='black')
                lab.plot(f, num.abs(t_er), color='red')
                lab.axvline(fmax/2., color='black')
                lab.axvline(fmax, color='gray')
                lab.xscale('log')
                lab.yscale('log')

                lab.gcf().add_subplot(2, 1, 2)
                lab.plot(f, num.angle(t_sx), color='black')
                lab.plot(f, num.angle(t_er), color='red')
                lab.xscale('log')
                lab.show()

                assert False, \
                    'evalresp and stationxml responses differ: %s' % str(nslc)


if __name__ == '__main__':
    util.setup_logging('test_fdsn', 'warning')
    unittest.main()
