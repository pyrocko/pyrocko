def dumb_parser( data ):
    
    (in_ws, in_kw, in_str) = (1,2,3)
    
    state = in_ws
    
    rows = []
    cols = []
    accu = ''
    for c in data:
        if state == in_ws:
            if c == '"':
                new_state = in_str
                
            elif c not in (' ', '\t', '\n', '\r'):
                new_state = in_kw
        
        if state == in_kw:
            if c in (' ', '\t', '\n', '\r'):
                cols.append(accu)
                accu = ''
                if c in ('\n','\r'):
                    rows.append(cols)
                    cols = []
                new_state = in_ws
                
        if state == in_str:
            if c == '"':
                accu += c
                cols.append(accu[1:-1])
                accu = ''
                if c in ('\n','\r'):
                    rows.append(cols)
                    cols = []
                new_state = in_ws
        
        state = new_state
    
        if state in (in_kw, in_str):
             accu += c
    if len(cols) != 0:
       rows.append( cols )
       
    return rows

pymseed.config.show_progress = False

class Programs:
    rdseed   = 'rdseed4.8'

def ensure_dir(d):
    if not os.path.isdir(d):
        if os.path.exists(d):
            sys.exit(d+' exists and is not a directory')
        os.mkdir( d )

def clean_dir(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    
    os.mkdir( d )

class Event:
    def __init__(self, lat, lon, time):
        self.lat = lat
        self.lon = lon
        self.time = time
        
class Station:
    def __init__(self, network, station, lat, lon, elevation, name='', components=None):
        self.network = network
        self.station = station
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.name = name
        if components is None:
            self.components = set()
        else:
            self.components = components
            
        self.dist_deg = None
        self.dist_m = None
        self.azimuth = None

    def set_event_relative_data( self, event ):
        self.dist_m = orthodrome.distance_accurate50m( event, self )
        self.dist_deg = self.dist_m / config.earthradius *orthodrome.r2d
        self.azimuth = orthodrome.azimuth(event, self)
        
    def __str__(self):
        return '%s.%s  %f %f %f  %f %f %f  %s' % (self.network, self.station, self.lat, self.lon, self.elevation, self.dist_m, self.dist_deg, self.azimuth, self.name)

class SeedVolumeAccess:

    def __init__(self, seedvolume, datapile=None):
        '''Create new SEED Volume access object.
        
        In:
            seedvolume -- filename of seed volume
            datapile -- if not None, this should be a pyrocko.pile.Pile object 
                with data traces which are then used instead of the data
                provided by the SEED volume. (This is useful for dataless SEED
                volumes.)
        '''
    
        self.seedvolume = seedvolume
        self.tempdir = tempfile.mkdtemp("","SeedVolumeAccess-")
        self._pile = datapile
        self._unpack()
        self._event = None
        self._stations = None

    def __del__(self):
        import shutil
        shutil.rmtree(self.tempdir)

    def iter_raw_traces(self):
        return self.get_pile().iter_all()

    def iter_displacement_traces(self, tfade, freqlimits, deltat=None):
        for trace in self.iter_raw_traces():
            try:
                if deltat is not None:
                    trace.downsample_to(deltat)
                
                respfile = pjoin(self.tempdir, 'RESP.%s.%s.%s.%s' % trace.nslc_id)
                trans = pymseed.InverseEvalresp(respfile, trace)

                displacement = trace.transfer(tfade, freqlimits, transfer_function=trans)
                
                yield displacement
            
            except pymseed.TraceTooShort:
                logging.warn('trace too short: %s' % trace)
            
            except pymseed.UnavailableDecimation:
                logging.warn('cannot downsample: %s' % trace)
                
    def get_pile(self):
        if self._pile is None:
            self._pile = pymseed.MSeedPile([ pjoin(self.tempdir, 'mini.seed') ] )
        return self._pile
        
    def get_event(self):
        if not self._event:
            self._event = self._get_events_from_file()[0]
        return self._event
        
    def get_stations(self):
        if not self._stations:
            self._stations = self._get_stations_from_file()
        
        event = self.get_event()
        
        for s in self._stations.values():
            s.set_event_relative_data(event)
            
        return self._stations
        
    def _unpack(self):
        input_fn = self.seedvolume
        output_dir = self.tempdir

        # seismograms:
        if self._pile is None:
            rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-d', '-z', '3', '-o', '4', '-p', '-R', '-q', output_dir], 
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out,err) = rdseed_proc.communicate()
            logging.info( 'rdseed: '+err )
        
        # event data:
        rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-e', '-q', output_dir], 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out,err) = rdseed_proc.communicate()
        logging.info( 'rdseed: '+err )
        
        # station summary information:
        rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-S', '-q', output_dir], 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out,err) = rdseed_proc.communicate()
        logging.info( 'rdseed: '+err )
        
        # station headers:
        rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-s', '-q', output_dir], 
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
        (out,err) = rdseed_proc.communicate()
        fout = open(os.path.join(output_dir,'station_header_infos'),'w')
        fout.write( out )
        fout.close()
        logging.info( 'rdseed: '+err )
        
    def _get_events_from_file( self ):
        rdseed_event_file =  os.path.join(self.tempdir,'rdseed.events')

        f = open(rdseed_event_file, 'r')
        events = []
        for line in f:
            toks = line.split(', ')
            if len(toks) > 4:
                datetime = toks[1].split('.')[0]
                lat = toks[2]
                lon = toks[3]
                format = '%Y/%m/%d %H:%M:%S'
                secs = calendar.timegm( time.strptime(datetime, format))
                e = Event(
                    lat = float(lat),
                    lon = float(lon),
                    time = secs
                )
                events.append(e)
                
        f.close()
        return events
            
    def _get_stations_from_file(self):
        rdseed_station_file = os.path.join(self.tempdir, 'rdseed.stations')
        
        f = open(rdseed_station_file, 'r')
        
        # sometimes there are line breaks in the station description strings
        
        txt = f.read()
        rows = dumb_parser( txt )
        f.close()
        
        icolname = 6
        icolcomp = 5
        
        stations = {}
        for cols in rows:
            s = Station(
                network = cols[1],
                station = cols[0],
                lat = float(cols[2]),
                lon = float(cols[3]),
                elevation = float(cols[4]),
                name = cols[icolname],
                components = set(cols[icolcomp].split())
            )
            stations[(s.network, s.station)] = s
                
        return stations
        
