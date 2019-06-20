# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division, print_function

import numpy as num
from pyrocko import model
from pyrocko.guts import Int, String, List

from .base import TargetGenerator

guts_prefix = 'pf.scenario'


class StationGenerator(TargetGenerator):
    nstations = Int.T(
        default=10,
        help='Number of randomly distributed stations.')

    network_name = String.T(
        default='CO',
        help='Network name')

    channels = List.T(
        optional=True,
        default=['BHE', 'BHN', 'BHZ'],
        help='Seismic channels to generate. Default: BHN, BHE, BHZ')


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

            if self.channels:
                channels = [model.station.Channel(c) for c in self.channels]
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

    def ensure_data(self, engine, sources, path, tmin=None, tmax=None):
        return []

    def add_map_artists(self, engine, sources, automap):
        automap.add_stations(self.get_stations())
