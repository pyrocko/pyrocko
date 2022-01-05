from __future__ import division, print_function, absolute_import

import unittest
import numpy as num
import logging
import os
import tempfile
from pyrocko import util, trace, evalresp
from pyrocko.io import stationxml
from pyrocko.client import fdsn, iris
from pyrocko.plot import response as plot_response

from .. import common

logger = logging.getLogger('pyrocko.test.test_fdsn')

stt = util.str_to_time

show_plot = int(os.environ.get('MPL_SHOW', 0))


def fix_resp_units(fn):
    with open(fn, 'r') as fin:
        with open(fn + '.temp', 'w') as fout:
            for line in fin:
                line = line.replace('count - ', 'counts - ')
                fout.write(line)

    os.unlink(fn)
    os.rename(fn + '.temp', fn)


class FDSNTestCase(unittest.TestCase):

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

    @common.require_internet
    @common.skip_on_download_error
    def test_availability(self):
        tmin = stt('2010-01-15 10:00:00')
        tmax = stt('2010-01-15 10:01:00')
        selection = [
            ('GE', 'EIL', '*', 'SHZ', tmin, tmax),
        ]

        xx = fdsn.availability('query', site='iris', selection=selection)
        assert xx.read().decode('utf8').splitlines()[1].startswith('GE')

        xx = fdsn.availability('extent', site='iris', selection=selection)
        assert xx.read().decode('utf8').splitlines()[1].startswith('GE')

    @common.require_internet
    @common.skip_on_download_error
    def test_check_params(self):
        for site in ['geofon']:
            with self.assertRaises(ValueError):
                fdsn.station(site=site,
                             network='GE',
                             station='EIL',
                             level='channel',
                             no_such_parameter=True)

    def test_read_big(self):
        for site in ['iris']:
            fpath = common.test_data_file('%s_1014-01-01_all.xml' % site)
            stationxml.load_xml(filename=fpath)

    @unittest.skipUnless(
        evalresp.have_evalresp(), 'evalresp not supported on this platform')
    @unittest.skip('needs manual inspection')
    @common.require_internet
    @common.skip_on_download_error
    def test_response(self, ntest=None):
        from pyrocko.evalresp_ext import EvalrespExtError

        tmin = stt('2014-01-01 00:00:00')
        tmax = stt('2014-01-02 00:00:00')
        # LD.PAL
        # SC.Y22A..UHZ
        sx = fdsn.station(
            site='iris',
            # network='II',
            # station='BFZ,ZKR',
            network='GE',
            channel='??Z',
            startbefore=tmin,
            endafter=tmax,
            level='channel', format='text')

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
                level='response')

            fi = iris.ws_resp(
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                tmin=tmin,
                tmax=tmax)

            fh, fn = tempfile.mkstemp()
            os.close(fh)
            with open(fn, 'wb') as fo:
                while True:
                    d = fi.read(1024)
                    if not d:
                        break

                    fo.write(d)

            fix_resp_units(fn)

            stages = (0, 100)
            resp_sx = sxr.get_pyrocko_response(
                nslc, timespan=(tmin, tmax),
                fake_input_units='M/S', stages=stages)

            # print(sxr)
            print(sxr.summary_stages())

            resp_er = trace.Evalresp(
                fn, target='vel', nslc_id=nslc, time=tmin)

            fmin = 0.001
            fmax = 100.

            er_gain_others = 1.0
            for _, _, channel in sxr.iter_network_station_channels(
                    net, sta, loc, cha, timespan=(tmin, tmax)):

                if channel.response:
                    fmax = channel.sample_rate.value * 0.5
                    for istage, stage in enumerate(
                            channel.response.stage_list):

                        if not (stages[0] <= istage < stages[1]):
                            er_gain_others *= stage.stage_gain.value

                break

            f = num.exp(num.linspace(num.log(fmin), num.log(fmax), 500))

            t_sx = resp_sx.evaluate(f)
            try:
                t_er = resp_er.evaluate(f) * er_gain_others
            except EvalrespExtError:
                continue

            abs_dif = num.abs(num.abs(t_sx) - num.abs(t_er)) / num.max(
                num.abs(t_er))

            mda = num.nanmean(abs_dif[f < 0.4*fmax])

            pha_dif = num.abs(num.angle(t_sx) - num.angle(t_er))

            mdp = num.nanmean(pha_dif[f < 0.4*fmax])

            print(mda, mdp)

            if mda > 1e-2 or mdp > 1e-4:
                from pyrocko.plot import graph_colors, to01
                styles = [
                    dict(
                        color=to01(graph_colors[i % len(graph_colors)]),
                        alpha=0.5)
                    for i in range(2)]

                plot_response.plot(
                    [resp_sx, resp_er],
                    fmin=fmin, fmax=fmax, nf=500,
                    labels=['StationXML', 'Evalresp'],
                    styles=styles)

                # lab.gcf().add_subplot(2, 1, 1)
                # lab.plot(f, num.abs(t_sx), color='black', lw=2.)
                # lab.plot(f, num.abs(t_er), color='red')
                # lab.axvline(fmax/2., color='black')
                # lab.axvline(fmax, color='gray')
                # lab.xscale('log')
                # lab.yscale('log')
                #
                # lab.gcf().add_subplot(2, 1, 2)
                # lab.plot(f, num.angle(t_sx), color='black', lw=2.)
                # lab.plot(f, num.angle(t_er), color='red')
                # lab.xscale('log')
                # if show_plot:
                #     lab.show()

            # assert mda < 0.05 and mdp < 0.04, \
            #     'evalresp and stationxml responses differ: %s' % str(nslc)

    @common.require_internet
    def test_wadl(self):
        for site in ['geofon']:  # fdsn.get_sites():
            logger.info(site)
            for service in ['dataselect', 'station', 'dataselect', 'event']:
                try:
                    wadl = fdsn.wadl(service, site=site, timeout=2.0)
                    str(wadl)
                    logger.info('  %s: OK' % (service,))

                except fdsn.Timeout:
                    logger.info('  %s: timeout' % (service,))

                except util.DownloadError:
                    logger.info('  %s: no wadl' % (service,))


if __name__ == '__main__':
    util.setup_logging('test_fdsn', 'warning')
    unittest.main()
