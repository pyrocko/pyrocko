import logging
import csv
import urlparse
import urllib2
import numpy as num
from datetime import datetime

from os import path
from ..trace import Trace  # noqa
from ..guts import Object, String, Float, Int, DateTimestamp, StringChoice


logger = logging.getLogger('pyrocko.gnss.ngl')


class StepEvent(Object):
    '''
    README http://geodesy.unr.edu/NGLStationPages/steps_readme.txt
    '''
    station_id = String.T(
        help='Station ID')
    time = DateTimestamp.T(
        help='Event time in isoformat')
    event = StringChoice.T(
        ['Equipment change', 'Possible earthquake'],
        help='Event type.'
             'Earthquake within 10^(0.5*mag - 0.8) degrees of station')
    USGS_eventid = String.T(
        default=None, 
        optional=True)


class GNSSStation(Object):
    station_id = String.T(
        help='Station ID')
    lat = Float.T(
        help='Latitude in WGS84')
    lon = Float.T(
        help='Longitude in WGS84')
    elevation = Float.T(
        help='Elevation in [m]')
    nsolutions = Int.T(
        help='Number of solutions')
    time_start = DateTimestamp.T(
        help='Time of acquisition start')
    time_end = DateTimestamp.T(
        help='Time of acquisition end')
    latency = StringChoice.T(
        ['2 Weeks', '24 Hours'],
        help='Solution latency')
    steps = []
    nsteps = 0


class NGL(object):
    url_2w = 'http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt'
    url_24h = ('http://geodesy.unr.edu/NGLStationPages/'
               'DataHoldingsRapid24hr.txt')
    url_steps = 'http://geodesy.unr.edu/NGLStationPages/steps.txt'

    url_data = 'http://geodesy.unr.edu/gps_timeseries/txyz/'
    reference_frame = 'IGS08'

    def __init__(self):
        self.stations = []
        self.nstations = 0
        self._read_station_list()

    def search_station(self, latitude, longitude, maxradius, minradius=0.):
        res = []
        for sta in self.stations:
            dist = ((sta.lat - latitude)**2 + (sta.lon - longitude)**2)**.5
            if dist <= maxradius and dist >= minradius:
                res.append(sta)
        return res

    def get_station(self, station_id):
        for station in self.stations:
            if station_id == station.station_id:
                return station
        raise ValueError('Station %s not found.' % station_id)

    def get_displacement(self, station_id):
        '''Get station displacement data

        :param station_id: Station ID, length of 4
        :type station_id: str
        :param starttime: Starttime, defaults to None
        :rtype: Station
        '''
        station = self.get_station(station_id)

        path = '{frame}/{station_id}.{frame}.txyz2'.format(
            frame=self.reference_frame,
            station_id=station_id)
        url = urlparse.urljoin(self.url_data, path)

        request = urllib2.Request(url)
        request.add_header('User-Agent', 'Pyrocko-GNSS_Client/0.1')
        data_txt = urllib2.urlopen(request, timeout=4)

        data = num.loadtxt(data_txt, usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        nsolutions = data.shape[0]

        print 'nsolutions ', nsolutions
        return data

    def _read_station_list(self):
        logger.debug('Initialising NGL database...')

        stations_24h = []
        with open(self._get_file(self.url_24h), 'r') as ngl_24h:
            ngl_24h.readline()
            for sta in ngl_24h:
                stations_24h.append(sta[0:4])

        with open(self._get_file(self.url_2w), 'r') as ngl_db:
            ngl_db.readline()
            stations = csv.reader(ngl_db, delimiter=' ', skipinitialspace=True)
            for sta in stations:
                try:
                    self.stations.append(
                        GNSSStation(
                            station_id=sta[0],
                            lat=float(sta[1]),
                            lon=float(sta[2]),
                            elevation=float(sta[3]),
                            nsolutions=int(sta[10]),
                            time_start=sta[7],
                            time_end=sta[8],
                            latency='2 Weeks' if sta[0] not in stations_24h
                                    else '24 Hours'))
                    self.stations[-1].regularize()
                except ValueError:
                    print logger.error('Could not read line: \'%s\'' % sta)
        self.nstations = len(self.stations)

    @staticmethod
    def _date_to_dec(date):
        return datetime.strptime('{}'.format(date),'%y%b%d').strftime('%Y-%m-%d')

    def get_event(self, station_id):
        events = []
        with open(self._get_file(self.url_steps), 'r') as ngl_steps:
            ngl_steps.readline()
            lines = csv.reader(ngl_steps, delimiter=' ', skipinitialspace=True)
            for line in lines:
                if line[0] == station_id:
                    print self._date_to_dec(line[1])
                    events.append(
                    StepEvent(
                        station_id=line[0], 
                        time=self._date_to_dec(line[1]),
                        event='Equipment change', 
                        ))
                    # if earthquake
                    if line[2] == 2: 
                        events[-1].event='Possible earthquake',
                        # distance from station to epicenter in km
                        events[-1].dist=line[4]
                        # event magnitude
                        events[-1].mag=line[5]
                        # USGS event ID
                        events[-1].USGS_eventid=line[6]
                    events[-1].regularize()
        return events

    @staticmethod
    def _get_file(url):
        from pyrocko import config

        filename = path.basename(url)
        file = path.join(config.config().cache_dir, filename)
        if not path.exists(file):
            logger.debug('Downloading %s' % url)
            from pyrocko import util
            util.download_file(url, file, None, None)
        else:
            logger.debug('Using cached %s' % filename)

        return file
