import logging

import orthodrome, trace, pile, config, model, eventdata, io, util
import os, sys, shutil, subprocess, tempfile, calendar, time, re

pjoin = os.path.join

logger = logging.getLogger('pyrocko.rdseed')

def cmp_version(a,b):
    ai = [ int(x) for x in a.split('.') ]
    bi = [ int(x) for x in b.split('.') ]
    return cmp(ai, bi)

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

class Programs:
    rdseed  = 'rdseed'
    checked = False

    @staticmethod
    def check():
        if not Programs.checked:
            try:
                rdseed_proc = subprocess.Popen([Programs.rdseed], 
                                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (out,err) = rdseed_proc.communicate()
            
            except OSError, e:
                if e.errno == 2:
                    reason =  "Could not find executable: '%s'." % Programs.rdseed
                else:
                    reason = str(e)
            
                logging.fatal('Failed to run rdseed program. %s' % reason)
                sys.exit(1)


            m = re.search(r'Release (\d+(\.\d+(\.\d+)?)?)', err)
            if not m:
                logger.warn('Cannot determine rdseed version number.')

            version = m.group(1)

            Programs.checked = True
            if cmp_version('4.7.5', version) == 1 or cmp_version(version, '5.0') == 1:
                logger.warn('Module pyrocko.rdseed has not been tested with version %s of rdseed.' % version)

class SeedVolumeNotFound(Exception):
    pass

class SeedVolumeAccess(eventdata.EventDataAccess):

    def __init__(self, seedvolume, datapile=None):
        
        '''Create new SEED Volume access object.
        
        In:
            seedvolume -- filename of seed volume
            datapile -- if not None, this should be a pyrocko.pile.Pile object 
                with data traces which are then used instead of the data
                provided by the SEED volume. (This is useful for dataless SEED
                volumes.)
        '''
        
        eventdata.EventDataAccess.__init__(self, datapile=datapile)
        self.tempdir = None
        Programs.check()


        self.tempdir = None
        self.seedvolume = seedvolume
        if not os.path.isfile(self.seedvolume):
            raise SeedVolumeNotFound()
        
        self.tempdir = tempfile.mkdtemp("","SeedVolumeAccess-")
        self.station_headers_file = os.path.join(self.tempdir, 'station_header_infos')
        self._unpack()

    def __del__(self):
        import shutil
        if self.tempdir:
            shutil.rmtree(self.tempdir)
                
    def get_pile(self):
        if self._pile is None:
            #fns = io.save( io.load(pjoin(self.tempdir, 'mini.seed')), pjoin(self.tempdir,
            #         'raw-%(network)s-%(station)s-%(location)s-%(channel)s.mseed'))
            fns = util.select_files( [ self.tempdir ], regex=r'\.SAC$')
            self._pile = pile.Pile()
            self._pile.load_files(fns, fileformat='sac')
            
        return self._pile
        
    def get_restitution(self, tr, allowed_methods):
        
        if 'evalresp' in allowed_methods:
            respfile = pjoin(self.tempdir, 'RESP.%s.%s.%s.%s' % tr.nslc_id)
            trans = trace.InverseEvalresp(respfile, tr)
            return trans
        else:
            raise eventdata.NoRestitution('no allowed restitution method available')
        
    def _unpack(self):
        input_fn = self.seedvolume
        output_dir = self.tempdir

        def strerr(s):
            return '\n'.join([ 'rdseed: '+line for line in s.splitlines() ])
        try:
            
            # seismograms:
            if self._pile is None:
                rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-d', '-z', '3', '-o', '1', '-p', '-R', '-q', output_dir], 
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                (out,err) = rdseed_proc.communicate()
                logging.info(strerr(err))
            
            # event data:
            rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-e', '-q', output_dir], 
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out,err) = rdseed_proc.communicate()
            logging.info(strerr(err) )
            
            # station summary information:
            rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-S', '-q', output_dir], 
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (out,err) = rdseed_proc.communicate()
            logging.info(strerr(err))
            
            # station headers:
            rdseed_proc = subprocess.Popen([Programs.rdseed, '-f', input_fn, '-s', '-q', output_dir], 
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
            (out,err) = rdseed_proc.communicate()
            fout = open(self.station_headers_file, 'w')
            fout.write( out )
            fout.close()
            logging.info(strerr(err))
        
        except OSError, e:
            if e.errno == 2:
                reason =  "Could not find executable: '%s'." % Programs.rdseed
            else:
                reason = str(e)
            
            logging.fatal('Failed to unpack SEED volume. %s' % reason)
            sys.exit(1)

    def _get_events_from_file( self ):
        rdseed_event_file =  os.path.join(self.tempdir,'rdseed.events')
        if not os.path.isfile(rdseed_event_file):
            return []
        
        f = open(rdseed_event_file, 'r')
        events = []
        for line in f:
            toks = line.split(', ')
            if len(toks) == 9:
                datetime = toks[1].split('.')[0]
                lat = toks[2]
                lon = toks[3]
                format = '%Y/%m/%d %H:%M:%S'
                secs = calendar.timegm( time.strptime(datetime, format))
                e = model.Event(
                    lat = float(toks[2]),
                    lon = float(toks[3]),
                    depth = float(toks[4])*1000.,
                    magnitude = float(toks[8]),
                    time = secs
                )
                events.append(e)
            else:
                raise Exception('Event description in unrecognized format')
            
        f.close()
        return events
    
    def _get_channel_orientations(self):
        
        orientations = {}
        pile = self.get_pile()
        
        for trace in pile.iter_all():
            dip = trace.meta['cmpinc']-90.
            azimuth = trace.meta['cmpaz']
            orientations[trace.nslc_id] = azimuth, dip
        return orientations
        
    def _get_stations_from_file(self):
        
        orientations = self._get_channel_orientations()
        
        # make station to locations map, cause these are not included in the 
        # rdseed.stations file
        
        p = self.get_pile()
        ns_to_l = {}
        for nslc in p.nslc_ids:
            ns = nslc[:2]
            if ns not in ns_to_l:
                ns_to_l[ns] = set()
            
            ns_to_l[ns].add(nslc[2])
        
        
        # make
        
        rdseed_station_file = os.path.join(self.tempdir, 'rdseed.stations')
        
        f = open(rdseed_station_file, 'r')
        
        # sometimes there are line breaks in the station description strings
        
        txt = f.read()
        rows = dumb_parser( txt )
        f.close()
        
        icolname = 6
        icolcomp = 5
        
        stations = []
        
        for cols in rows:
            network, station = cols[1], cols[0]
            for location in ns_to_l[network, station]:
                
                channels = []
                for channel in  cols[icolcomp].split():
                    if (network, station, location, channel) in orientations:
                        azimuth, dip = orientations[network, station, location, channel]
                    else:
                        azimuth, dip = None, None # let Channel guess from name
                    channels.append(model.Channel(channel, azimuth, dip))
                    
                s = model.Station(
                    network = cols[1],
                    station = cols[0],
                    location = location,
                    lat = float(cols[2]),
                    lon = float(cols[3]),
                    elevation = float(cols[4]),
                    name = cols[icolname],
                    channels = channels
                )
                stations.append(s)
                
        return stations
        
        
    def _insert_channel_descriptions(self, stations):
        # this is done beforehand in this class
        pass
    
