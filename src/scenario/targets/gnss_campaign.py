import logging
import os.path as op
import numpy as num

from pyrocko import gf, util
from pyrocko.guts import Float

from .base import TargetGenerator, NoiseGenerator
from .station import RandomStationGenerator, StationGenerator

DEFAULT_STORE_ID = 'ak135_static'

logger = logging.getLogger('pyrocko.scenario.targets.gnss_campaign')
guts_prefix = 'pf.scenario'


class GPSNoiseGenerator(NoiseGenerator):
    measurement_duarion_days = Float.T(
        default=2.,
        help='Measurement duration in length')


class GNSSCampaignGenerator(TargetGenerator):
    station_generator = StationGenerator.T(
        default=RandomStationGenerator.D(),
        help='The StationGenerator for creating the stations.')

    noise_generator = NoiseGenerator.T(
        default=GPSNoiseGenerator.D(),
        help='Add Synthetic noise to the GNSS displacements.')

    store_id = gf.StringID.T(
        default=DEFAULT_STORE_ID,
        help='The GF store to use for forward-calculations.')

    def get_stations(self):
        return self.station_generator.get_stations()

    def get_targets(self):
        stations = self.get_stations()
        lats = num.array([s.lat for s in stations])
        lons = num.array([s.lon for s in stations])

        target = gf.GNSSCampaignTarget(
            lats=lats,
            lons=lons,
            store_id=self.store_id)

        return [target]

    def get_gnss_campaign(self, engine, sources, tmin=None, tmax=None):
        resp = engine.process(
            sources,
            self.get_targets(),
            nthreads=0)

        campaigns = [r.campaign for r in resp.static_results()]

        stacked_campaign = campaigns[0]
        for camp in campaigns[1:]:
            for ista, sta in enumerate(camp.stations):
                stacked_campaign.stations[ista].north += sta.north
                stacked_campaign.stations[ista].east += sta.east
                stacked_campaign.stations[ista].up += sta.up

        for ista, sta in enumerate(stacked_campaign.stations):
            sta.code = 'SY%02d' % (ista + 1)

        return [stacked_campaign]

    def dump_data(self, engine, sources, path,
                  tmin=None, tmax=None, overwrite=False):
        path_gnss = op.join(path, 'gnss')
        util.ensuredir(path_gnss)

        campaigns = self.get_gnss_campaign(
            engine, sources, tmin, tmax)

        fn = op.join(path_gnss, 'campaign.yml')

        with open(fn, 'w') as f:
            for camp in campaigns:
                camp.dump(stream=f)

        return [fn]
