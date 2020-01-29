# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division, print_function

import numpy as num
from pyrocko import model
from pyrocko.guts import Int, String, List

from pyrocko.model.station import load_stations
from pyrocko.io import stationxml
from pyrocko.orthodrome import distance_accurate50m

from .base import LocationGenerator

guts_prefix = 'pf.scenario'


class StationGenerator(LocationGenerator):

    def get_stations(self):
        raise NotImplementedError

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


class ImportStationGenerator(StationGenerator):

    stations_paths = List.T(
        optional=True,
        help='List of files with station coordinates in Pyrocko format.')

    stations_stationxml_paths = List.T(
        optional=True,
        help='List of files with station coordinates in StationXML format.')

    def __init__(self, **kwargs):
        StationGenerator.__init__(self, **kwargs)
        self._stations = None

    def has_stations(self):
        if not self.get_stations():
            return False
        return True

    def get_center_latlon(self):
        stations = self.get_stations()
        if not stations:
            return self._parent.get_center_latlon()

        lats = [st.lat for st in stations]
        lons = [st.lon for st in stations]
        lat_min = min(lats)
        lat_max = max(lats)
        lon_min = min(lons)
        lon_max = max(lons)

        return (lat_max - lat_min)/2, (lon_max - lon_min)/2

    def get_radius(self):
        stations = self.get_stations()
        if not stations:
            return 10000.

        lats = [st.lat for st in stations]
        lons = [st.lon for st in stations]
        lat_min = min(lats)
        lat_max = max(lats)
        lon_min = min(lons)
        lon_max = max(lons)

        return distance_accurate50m(lat_min, lon_min, lat_max, lon_max) / 2.

    def get_stations(self):
        if self._stations is None:

            stations = []

            if self.stations_paths:
                for filename in self.stations_paths:
                    stations.extend(
                        load_stations(filename))

            if self.stations_stationxml_paths:
                for filename in self.stations_stationxml_paths:
                    sxml = stationxml.load_xml(filename=filename)
                    stations.extend(
                        sxml.get_pyrocko_stations())

            self._stations = stations

        return self._stations

    def nsl(self, istation):
        stations = self.get_stations()
        return stations[istation].nsl()

    def clear(self):
        self._stations = None

    @property
    def nstations(self):
        return len(self.get_stations())


class RandomStationGenerator(StationGenerator):

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
                lat, lon, north_shift, east_shift, depth = map(
                    float, self.get_coordinates(istation))

                net, sta, loc = self.nsl(istation)
                station = model.Station(
                    net, sta, loc,
                    lat=lat,
                    lon=lon,
                    north_shift=north_shift,
                    east_shift=east_shift,
                    depth=depth,
                    channels=channels)

                stations.append(station)

            self._stations = stations

        return self._stations
