import logging

import orthodrome, trace, pile, config, model
import os, sys, shutil, subprocess

logger = logging.getLogger('pyrocko.rdseed')

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
    rdseed   = 'rdseed4.8'



class SeedVolumeAccess(EventDataAccess):

    def __init__(self, seedvolume, datapile=None):
        
        '''Create new SEED Volume access object.
        
        In:
            seedvolume -- filename of seed volume
            datapile -- if not None, this should be a pyrocko.pile.Pile object 
                with data traces which are then used instead of the data
                provided by the SEED volume. (This is useful for dataless SEED
                volumes.)
        '''
    
        EventDataAccess.__init__(self, datapile=datapile)
    
        self.seedvolume = seedvolume
        self.tempdir = tempfile.mkdtemp("","SeedVolumeAccess-")
        self._unpack()

    def __del__(self):
        import shutil
        shutil.rmtree(self.tempdir)
                
    def get_pile(self):
        if self._pile is None:
            self._pile = pile.Pile([ pjoin(self.tempdir, 'mini.seed') ] )
        return self._pile
        
    def get_restitution(self, tr):
        respfile = pjoin(self.tempdir, 'RESP.%s.%s.%s.%s' % tr.nslc_id)
        trans = trace.InverseEvalresp(respfile, tr)
        return trans
        
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
                e = model.Event(
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
        
        stations = []
        for cols in rows:
            s = model.Station(
                network = cols[1],
                station = cols[0],
                location = '*',
                lat = float(cols[2]),
                lon = float(cols[3]),
                elevation = float(cols[4]),
                name = cols[icolname],
                components = set(cols[icolcomp].split())
            )
            stations.append[s]
                
        return stations
        
