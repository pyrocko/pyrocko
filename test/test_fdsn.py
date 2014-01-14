import unittest
import os
from pyrocko import fdsn
from pyrocko import util


def tts(t):
    if t is not None:
        return util.time_to_str(t)
    else:
        return '...'

stt = util.str_to_time


class FDSNStationTestCase(unittest.TestCase):

    def test_read_samples(self):
        ok = False
        for fn in ['geeil.iris.xml', 'geeil.geofon.xml']:
            fpath = self.file_path(fn)
            x = fdsn.station.load_xml(filename=fpath)
            for network in x.network_list:
                assert network.code == 'GE'
                for station in network.station_list:
                    assert station.code == 'EIL'
                    for channel in station.channel_list:
                        assert channel.code[:2] == 'BH'
                        for stage in channel.response.stage_list:
                            ok = True

            assert ok

            assert len(x.get_pyrocko_stations()) in (3, 4)
            for s in x.get_pyrocko_stations():
                assert len(s.get_channels()) == 3

            assert len(x.get_pyrocko_stations(
                time=stt('2010-01-15 10:00:00'))) == 1

    def test_retrieve(self):
        for site in ['geofon', 'iris']:
            fsx = fdsn.ws.station(site=site,
                                  network='GE', 
                                  station='EIL',
                                  level='channel')

            assert len(fsx.get_pyrocko_stations(
                time=stt('2010-01-15 10:00:00'))) == 1

    def file_path(self, fn):
        return os.path.join(os.path.dirname(__file__), 'stationxml', fn)

    def test_retrieve_big(self):
        for site in ['iris']:
            fpath = self.file_path('%s_1014-01-01_all.xml' % site)

            if not os.path.exists(fpath):
                with open(fpath, 'w') as f:
                    source = fdsn.ws.station(
                        site=site,
                        startbefore=stt('2014-01-01 00:00:00'),
                        endafter=stt('2014-01-02 00:00:00'),
                        level='station', parsed=False, matchtimeseries=True)

                    while True:
                        data = source.read(1024)
                        if not data:
                            break
                        f.write(data)

            fsx = fdsn.station.load_xml(filename=fpath)
            for station in fsx.get_pyrocko_stations():
                print station

            print len(fsx.get_pyrocko_stations())

import time
gt = None
def lap():
    global gt
    t = time.time()
    if gt is not None:
        diff = t - gt
    else:
        diff = 0

    gt = t
    return diff


if __name__ == '__main__':
    util.setup_logging('test_fdsn_station', 'warning')
    unittest.main()
