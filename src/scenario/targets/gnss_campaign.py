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

    def add_noise(self, campaign):
        # https://www.nat-hazards-earth-syst-sci.net/15/875/2015/nhess-15-875-2015.pdf
        waterlevel = 1. - (.99 + .0015 * self.measurement_duarion_days)

        for ista, sta in enumerate(campaign.stations):
            rstate = self.get_rstate(ista)

            sta.north.error = 2e-5
            sta.east.error = 2e-5

            sta.north.shift += rstate.normal(0., sta.north.error)
            sta.east.shift += rstate.normal(0., sta.east.error)


class GNSSCampaignGenerator(TargetGenerator):
    station_generator = StationGenerator.T(
        default=RandomStationGenerator(
            network_name='GN'),
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
                stacked_campaign.stations[ista].north.shift += sta.north.shift
                stacked_campaign.stations[ista].east.shift += sta.east.shift
                stacked_campaign.stations[ista].up.shift += sta.up.shift

        for ista, sta in enumerate(stacked_campaign.stations):
            sta.code = 'SY%02d' % (ista + 1)

        if self.noise_generator is not None:
            self.noise_generator.add_noise(stacked_campaign)
            pass

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
