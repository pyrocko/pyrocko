import math
import numpy as num
import pyrocko.orthodrome as od

from pyrocko.guts import (Object, Float, String, List, StringChoice,
                          DateTimestamp)
from pyrocko.model import Location

guts_prefix = 'pf.gnss'


class GNSSComponent(Object):
    ''' Component of a GNSSStation
    '''
    unit = StringChoice.T(
        choices=['mm', 'cm', 'm'],
        default='m')

    shift = Float.T(
        default=0.,
        help='Shift in unit')

    sigma = Float.T(
        default=0.,
        help='One sigma uncertainty of the measurement')

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise AttributeError('Other has to be of instance %s'
                                 % self.__class__)
        comp = self.__class__()
        comp.shift = self.shift + other.shift
        comp.sigma = math.sqrt(self.sigma**2 + other.sigma**2)
        return comp

    def __iadd__(self, other):
        self.shift += other.shift
        self.sigma = math.sqrt(self.sigma**2 + other.sigma**2)
        return self


class GNSSStation(Location):
    ''' Representation of a GNSS station during a campaign measurement

    For more information see
    http://kb.unavco.org/kb/assets/660/UNAVCO_Campaign_GPS_GNSS_Handbook.pdf
    '''

    code = String.T(
        help='Four letter station code',
        optional=True)

    style = StringChoice.T(
        choices=['static', 'rapid_static', 'kinematic'],
        default='static')

    survey_start = DateTimestamp.T(
        optional=True)

    survey_end = DateTimestamp.T(
        optional=True)

    north = GNSSComponent.T(
        default=GNSSComponent.D())

    east = GNSSComponent.T(
        default=GNSSComponent.D())

    up = GNSSComponent.T(
        default=GNSSComponent.D())

    def __init__(self, *args, **kwargs):
        Location.__init__(self, *args, **kwargs)


class GNSSCampaign(Object):

    stations = List.T(
        GNSSStation.T(),
        help='List of GNSS campaign measurements')

    name = String.T(
        help='Campaign name',
        default='Unnamed campaign')

    survey_start = DateTimestamp.T(
        optional=True)

    survey_end = DateTimestamp.T(
        optional=True)

    def add_station(self, station):
        return self.stations.append(station)

    def get_station(self, station_code):
        for sta in self.stations:
            if sta.code == station_code:
                return sta
        raise ValueError('Could not find station %s' % station_code)

    def get_center_latlon(self):
        return od.geographic_midpoint_locations(self.stations)

    def get_radius(self):
        coords = self.coordinates
        ll = coords[:, 0].min(), coords[:, 1].min()
        ur = coords[:, 0].max(), coords[:, 1].max()
        return od.distance_accurate50m(*ll, *ur) / 2.

    @property
    def coordinates(self):
        return num.array([loc.effective_latlon for loc in self.stations])

    @property
    def nstations(self):
        return len(self.stations)
