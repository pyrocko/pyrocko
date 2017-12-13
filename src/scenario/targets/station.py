import numpy as num
from pyrocko import model
from pyrocko.guts import Int, String, Bool

from .base import TargetGenerator

guts_prefix = 'pf.scenario'


class StationGenerator(TargetGenerator):
    nstations = Int.T(
        default=10,
        help='Number of randomly distributed stations.')

    network_name = String.T(
        default='CO',
        help='Network name')

    with_channels = Bool.T(
        default=True,
        help='Generate seismic channels: BHN, BHE, BHZ')


class RandomStationGenerator(StationGenerator):

    def __init__(self, **kwargs):
        StationGenerator.__init__(self, **kwargs)
        self._stations = None

    def clear(self):
        StationGenerator.clear(self)
        self._stations = None

    def nsl(self, istation):
        return self.network_name, 'S%03i' % (istation + 1), '',

    def get_stations(self):
        if self._stations is None:

            if self.with_channels:
                channels = []
                for comp in ('BHN', 'BHE', 'BHZ'):
                    channels.append(model.station.Channel(comp))
            else:
                channels = None

            stations = []
            for istation in range(self.nstations):
                lat, lon = self.get_latlon(istation)

                net, sta, loc = self.nsl(istation)
                station = model.Station(
                    net, sta, loc,
                    lat=lat,
                    lon=lon,
                    channels=channels)

                stations.append(station)

            self._stations = stations

        return self._stations

    def get_distance_range(self, sources):
        dists = []
        for source in sources:
            for station in self.get_stations():
                dists.append(
                    source.distance_to(station))

        return num.min(dists), num.max(dists)

    def dump_data(self, engine, sources, path, *args, **kwargs):
        return []

    def add_map_artists(self, engine, sources, automap):
        automap.add_stations(self.get_stations())
