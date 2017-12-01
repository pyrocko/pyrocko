from pyrocko import model
from pyrocko.guts import Int

from .base import TargetGenerator

guts_prefix = 'pf.scenario'


class StationGenerator(TargetGenerator):
    pass


class RandomStationGenerator(StationGenerator):

    nstations = Int.T(
        default=10,
        help='Number of randomly distributed stations.')

    def __init__(self, **kwargs):
        StationGenerator.__init__(self, **kwargs)
        self._stations = None

    def clear(self):
        StationGenerator.clear(self)
        self._stations = None

    def nsl(self, istation):
        return '', 'S%03i' % (istation + 1), '',

    def get_stations(self):
        if self._stations is None:
            stations = []
            for istation in range(self.nstations):
                lat, lon = self.get_latlon(istation)

                net, sta, loc = self.nsl(istation)
                station = model.Station(
                    net, sta, loc,
                    lat=lat,
                    lon=lon)

                stations.append(station)

            self._stations = stations

        return self._stations

    def dump_stations(self):
        pass
