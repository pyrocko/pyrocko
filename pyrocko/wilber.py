import urllib, urllib2, re, sys
import calendar, time
import logging
import random

def strgmtime(secs):
    return time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(secs))

def intersect(a,b):
    return a[1] > b[0] and b[1] >= a[0]

def interval_and(a,b):
    return (max(a[0],b[0]), min(a[1],b[1]))

class Event:
    def __init__(self, timestamp, mag, lat, lon, depth, region, datasource, urlend):
        self.timestamp = timestamp
        self.mag = mag
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.region = region
        self.datasource = datasource
        self.urlend = urlend
        
    def __str__(self):
        return '%s %6s %3.1f %6.2f %7.2f %5.1f %s' % (
            strgmtime(self.timestamp),
            self.datasource, self.mag, self.lat, self.lon, self.depth, self.region
        )

class Station:
    def __init__(self, station, network, dist, azimuth, channels, snr):
        self.station = station
        self.network = network
        self.dist = dist
        self.azimuth = azimuth
        self.channels = channels
        self.snr = snr
        
    def __str__(self):
        return '%s %s %3.0f %3.0f %s %g' % (self.network, self.station, self.dist, self.azimuth, self.channels, self.snr)

def to_secs(date, time):
    toks = date.split('/')
    toks.extend(time.split(':'))
    toks[-1] = round(float(toks[-1]))
    return calendar.timegm([ int(x) for x in toks ])
    
class WilberRequestError(Exception):
    pass
    
class Wilber:
    def get_events(time_range = None):
        raise Exception('This method should be implemented in derived class.')
    
    def get_data( event,
                  wanted_channels = ('BHE', 'BHN', 'BHZ'),
                  before = 5,
                  after = 50,
                  outfilename='event_data.seed'):
        
        raise Exception('This method should be implemented in derived class.')
    
    def event_filter(self, event):
        return True
    
    def station_filter(self, station):
        return True
    
class OrfeusWilber(Wilber):
    pass

class IrisWilber(Wilber):
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.urlbeg = 'http://www.iris.edu'
        # cgidir = '/cgi-bin'            # old url
        cgidir = '/cgi-bin/wilberII'     # new url
        self.urlend1 = cgidir + '/wilberII_page1.pl'
        self.urlend2 = cgidir + '/wilberII_page2.pl'
        self.urlend3 = cgidir + '/wilberII_page3.pl'
        self.urlend4 = cgidir + '/wilberII_page4.pl'
        
    def extract_stations(self, page):
        r = re.compile(r'station\(([^)]+)\)')
        stations = []
        for str in r.findall(page):
            toks = str.replace("'",'').replace(',,',',').split(',')
            if toks[0] == 'name': continue
            st = Station( station=toks[0],
                          network=toks[1],
                          dist=float(toks[2]),
                          azimuth=float(toks[3]),
                          channels=toks[4:-1],
                          snr=float(toks[-1]) )
            
            stations.append(st)
    
        return stations
            
    def extract_events(self, page):
        pat = r'color=([0-9a-f]+).+href="(' + re.escape(self.urlend3) + '\?[^"]+)">([^<]+)</a>([^<]+)</font>'
        r = re.compile(pat)
        iev = 0
        old_color = ''
        gather = []
        events = []
        for line in page.splitlines():
            m = r.findall(line)
            if len(m) == 1:
                color, urlend, datetime, descr = m[0]
                if color != old_color:
                    old_color = color
                    if gather:
                        events.append(gather)
                    gather = []
                    
                toks = descr.split(None,5)
                assert len(toks)==6
                
                timestamp = to_secs(*datetime.split())
                
                datasource, mag, lat, lon, depth, region = toks
                for s in 'SPYDER', 'FARM':
                    if datasource.startswith(s):
                        datasource = s
                        
                mag = float(mag)
                lat = float(lat)
                lon = float(lon)
                depth = float(depth)
                ev = Event(timestamp, mag, lat, lon, depth, region, datasource, urlend)
                gather.append(ev)
        
        if gather:
            events.append(gather)
        
        return events
    
    def check_request_error(self, page):
        r = re.compile(r'<b>REQUEST ERROR: ([^<]+)<')
        m = r.search(page)
        if m:
            raise WilberRequestError(m.group(1))

    def extract_hidden_params(self, page):
        r = re.compile(r'type=hidden\s+name=(\S+)\s+value=([^>]+)>')
        hidden_params = []
        for name,value in r.findall(page):
            hidden_params.append((name, value.strip('"')))
            
        return hidden_params
    
    def extract_status_page_link(self, page):
        r = re.compile(r'<a href="([^"]+)">next</a>')
        return r.findall(page)[0]
    
    def extract_ftp_link(self, page):
        r = re.compile(r'<a href="(ftp://[^"]+)">')
        res = r.findall(page)
        if res:
            return res[0]
        else:
            return None
        
    def get_time_intervals(self):
        
        page = urllib2.urlopen(self.urlbeg+self.urlend1).read()
        
        r = re.compile(r'VALUE="(Q(\d)_(\d\d\d\d))"')
        quartals = r.findall(page)
        intervals = []
        for label, q, year in quartals:
            q = int(q)
            year = int(year)
            month = (q-1)*3+1
            tmin = calendar.timegm((year,month,1,0,0,0))
            month += 3
            if month == 13:
                year += 1
                month = 1
            tmax = calendar.timegm((year,month,1,0,0,0))
            intervals.append( (tmin, tmax, label) )
        
        intervals.sort()
        return intervals
        
    def get_relevant_time_intervals(self, time_range):
        
        relevant = []
        intervals = self.get_time_intervals()
        
        tmaxmax = time_range[0]
        for tmin, tmax, label in intervals:
            tmaxmax = max(tmax, tmaxmax)
            if intersect((tmin,tmax),time_range):
                tmi, tma = interval_and((tmin,tmax), time_range)
                relevant.append( (tmi, tma, label) )
                
        if time_range[1] >= tmaxmax:
            tmi, tma = interval_and((tmaxmax,time_range[1]), time_range)
            relevant.append( (tmi, tma, 'current') )
        
        relevant.sort()
        return relevant
        
    def get_events(self, time_range = None):
        
        if time_range is None: # by default, get events in past 24 hours
            now = time.time()
            time_range = (now-24*60*60, now)
        
        xevents = []
        for tmi, tma, lab in self.get_relevant_time_intervals(time_range):
            logging.info('Querying event list page (%s).' % lab)
            params = {'event_map': lab, 'radius': 'all'}
            eparams = urllib.urlencode(params)
            page = urllib2.urlopen(self.urlbeg+self.urlend2, eparams).read()
            events = self.extract_events(page)
            for event_group in events:
                event_group_filtered = [ ev for ev in event_group if self.event_filter(ev) and
                                         tmi <= ev.timestamp and ev.timestamp < tma ]
                
                if event_group_filtered:
                    event_group_filtered.sort( lambda a,b: cmp(a.datasource, b.datasource))
                    xevents.append(event_group_filtered[0])
                    
        nev = len(xevents)
        if nev == 0:
            logging.warn('No events matching given criteria found.')
        elif nev == 1:
            logging.info('Found one distinct event matching given criteria.')
        else:
            logging.info('Found %i distinct events matching given criteria.' % nev)

        return xevents
        
    
    def get_data(self, event,
                      wanted_channels = ('BHE', 'BHN', 'BHZ'),
                      vnetcodes = ('_GSN-BROADBAND',),
                      netcodes = (),
                      before = 5,
                      after = 50,
                      outfilename='event_data.seed'):
            
            logging.info('Attempt to download data for event %s' % str(event))
            
            logging.info('Querying network selection page.')
            page = urllib2.urlopen(self.urlbeg+event.urlend).read()
            
            params = self.extract_hidden_params(page)
            for vnetcode in vnetcodes:
                params.append(('vnetcode', vnetcode))
            
            for netcode in netcodes:
                params.append(('netcode', netcode))
            
            eparams = urllib.urlencode(params)
            
            logging.info('Querying station selection page.')
            page = urllib2.urlopen(self.urlbeg+self.urlend4, eparams).read()
            
            hidden_params = self.extract_hidden_params(page)
            stations = self.extract_stations(page)
            
            stations_filtered = []
            for st in stations:
                take = False
                for comp in wanted_channels:
                    if comp in st.channels:
                        take = True
                if take and self.station_filter(st):
                    stations_filtered.append(st)
            
            nstations = len(stations_filtered)
            if nstations == 0:
                logging.warn('No events matching given criteria found.')
                return 
            
            logging.info('Number of stations selected: %i' % len(stations_filtered))
                        
            params = []
            
            hidden_params = dict(hidden_params)
            for k in ('evname', 'qtrxyrad', 'vnetcode'):
                params.append( (k, hidden_params[k]) )
                
            params.append( ('network', 'ALL') )
            
            for comp in wanted_channels:
                params.append(('channel', comp))
                
            for st in stations_filtered:
                params.append(('S%s_%s' % (st.station, st.network),
                               '%s_%s' % (st.station, st.network)))
            
            
            label = 'event_%i_%i' % (event.timestamp, random.randint(1000000,9999999))
            
            params.extend([
                ('data_format', 'SEED'),
                ('before', before),
                ('after', after),
                ('username', self.username),
                ('label', label),
                ('email', self.email),
                ('request', 'Process Request')
            ])
                        
            eparams = urllib.urlencode(params)
            
            logging.info('Requesting data...')
            req = urllib2.urlopen(self.urlbeg+self.urlend4, eparams)
            
            logging.info('Waiting for response...')
            page = req.read()
            
            self.check_request_error(page)
            
            statuspage = self.extract_status_page_link(page)
            
            url = self.urlbeg+statuspage
            
            logging.info('Status page URL is: %s' % url)
            logging.info('Waiting for data to become ready on FTP server.')
            
            while True:
                page = urllib2.urlopen(url).read()
                ftplink = self.extract_ftp_link(page)
                if ftplink is not None:
                    break
                
                logging.info('(waiting...)')
                time.sleep(10)
                
            ftplink += '/%s.seed' % label
    
            logging.info('Data is available on FTP server.')
            logging.info('FTP URL is: %s' % ftplink)
            
            out = open(outfilename, 'w')
            logging.info('Connecting to FTP server...')
            ftpcon = urllib2.urlopen(ftplink)
            
            l = 0
            while True:
                data = ftpcon.read(1024*1024)
                n = len(data)
                if n == 0:
                    break
                l += n
                logging.info('(downloading...) %i B downloaded.' % l)
                out.write(data)
            
            ftpcon.close()
                
            out.close()
            logging.info('Download complete.')


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO,)

    # Example: Get data for all events in Jan 2007, having a magnitude > 6 and
    #          a depth of less than 50 km.
    #          Get that data only for stations from GSN-BROADBAND at epicentral
    #          distances up to 90 deg, and only channels (BHZ, BHE, BHN).

    class GlobalDataRequest(IrisWilber):
        
        def station_filter(self, station):
            #return station.dist < 10.
            return True
            
        def event_filter(self, event):
            return  True
    
    request = GlobalDataRequest(username='sebastian')
    #tr = ( calendar.timegm((2008,1,1,0,0,0)),
    #       calendar.timegm((2009,1,1,0,0,0)) )
    
    now = time.time()
    tr = ( now - 24*60*60 * 2, now - 24*60*60 )
    
    events = request.get_events(time_range=tr)# time_range=jan_2007)
    
    for event in events:
        print event
    
    for event in events:
        outfilename = 'data_%i.seed' % event.timestamp
        request.get_data(events[-1], outfilename=outfilename, vnetcodes=['_ILAR'], netcodes=['GE'] )



