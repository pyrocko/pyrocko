
from pyrocko import util, model, trace
import sys, re, calendar, time, logging
import numpy as num

unpack_fixed = util.unpack_fixed

logger = logging.getLogger('pyrocko.gse2')

km = 1000.

instrument_descriptions = {
'Akashi': 'Akashi',
'20171A': 'Geotech 20171A',
'23900': 'Geotech 23900',
'7505A': 'Geotech 7505A',
'8700C': 'Geotech 8700C',
'BB-13V': 'Geotech BB-13V',
'CMG-3': 'Guralp CMG-3',
'CMG-3N': 'Guralp CMG-3NSN',
'CMG-3T': 'Guralp CMG-3T',
'CMG-3E': 'Guralp CMG3-ESP',
'FBA-23': 'Kinemetrics FBA-23',
'GS-13': 'Geotech GS-13',
'GS-21': 'Geotech GS-21',
'HM-500': 'HM-500',
'KS3600': 'Geotech KS-36000',
'KS360i': 'Geotech KS-36000-I',
'KS5400': 'Geotech KS-54000',
'LE-3D': 'LE-3D',
'Mk II': 'Willmore Mk II',
'MP-L4C': 'Mark Products L4C',
'Oki': 'Oki',
'Parus2': 'Parus-2',
'Podrst': 'Podrost',
'S-13': 'Geotech S-13',
'S-500': 'Geotech S-500',
'S-750': 'Geotech S-750',
'STS-1': 'Streckeisen STS-1',
'STS-2': 'Streckeisen STS-2',
'SDSE-1': 'SDSE-1',
'SOSUS': 'SOSUS',
'TSJ-1e': 'TSJ-1e'}

modulus = 100000000
def checksum_slow(data):
    checksum = 0
    for x in data:
        checksum += x % modulus
        checksum = checksum % modulus
    return abs(checksum)    



def isd(line, toks, name, nargs=None):
    if not line.lower().startswith(name.lower()):
        return False
    if isinstance(nargs, int):
        if len(toks) != nargs:
            return False
    if isinstance(nargs, tuple):
        nmin, nmax = nargs
        if nmin is not None and len(toks) < nmin:
            return False
        if nmax is not None and len(toks) > nmax:
            return False
    return True

def slashdate(s):
    if s == '':
        return None
    else:
        return calendar.timegm(time.strptime(s, '%Y/%m/%d'))

def sslashdate(i):
    if i is None:
        return '          '
    else:
        return time.strftime('%Y/%m/%d', time.gmtime(i))

class Anon:
    pass

class GSE:
    def __init__(self):
        self.version = None
        self.msg_type = None
        self.msg_id = None
        self.ref_id = None
        self.waveforms = []
        self.error_logs = []
        self.stations = []
        self.channels = []
    
    def add(self, content):
        if isinstance(content, Waveform):
            self.waveforms.append(content)
        if isinstance(content, ErrorLog):
            self.error_logs.append(content)
        if isinstance(content, Station):
            self.stations.append(content)
        if isinstance(content, Channel):
            self.channels.append(content)
            
    def __str__(self):
        s = []
        for waveform in self.waveforms:
            s.append(str(waveform))

        s.append('')
        return '\n'.join(s)

    def get_pyrocko_stations(self):
        from_stations = {}
        for gs in self.stations:
            ps = model.Station(
                network = gs.network,
                station = gs.station,
                location = '',
                lat = gs.lat,
                lon = gs.lon,
                elevation = gs.elevation*km)
                
            from_stations[gs.network, gs.station] = ps
        
        from_channels = {}
        for gc in self.channels:
            nsl = gc.network, gc.station, gc.auxid
            if nsl not in from_channels:
                ps = model.Station(
                    network = gc.network,
                    station = gc.station,
                    location = gc.auxid,
                    lat = gc.lat,
                    lon = gc.lon,
                    elevation = gc.elevation*km,
                    depth = gc.depth*km)
                from_channels[nsl] = ps
                
            pc = model.Channel(name=gc.channel, azimuth=gc.hang, dip=gc.vang-90.0)
            from_channels[nsl].add_channel(pc)
        
        for nsl, ps in from_channels.iteritems():
            ns = nsl[:2]
            if ns in from_stations:
                del from_stations[ns]
            
        return from_stations.values() + from_channels.values()
            
class Waveform:
    def __init__(self, wid2, sta2, chk2, dat2):
        self.wid2 = wid2
        self.sta2 = sta2
        self.chk2 = chk2
        self.dat2 = dat2
      
        assert self.wid2.sub_format in 'INT CM6 CM8 AUT AU6 AU8'.split()
        if dat2.rawdata is not None:
            if self.wid2.sub_format == 'CM6':
                from pyrocko import gse2_ext
                self.data = gse2_ext.decode_m6( ''.join(dat2.rawdata), wid2.samps )
            else:
                logger.error('cannot load GSE sub format "%s" (not implemented)' % self.wid2.sub_format) 
                self.data = None

        else:
            self.data = None

        if self.data is None:
            self.tmax = self.wid2.tmin + 1.0/self.wid2.samprate * (wid2.samps - 1)
        else:
            self.tmax = None


    def __str__(self):
        return ' '.join([self.wid2.station, self.wid2.channel, self.wid2.auxid, self.wid2.sub_format, util.gmctime(self.wid2.tmin)])

    def trace(self):
        return trace.Trace(station=self.wid2.station, location=self.wid2.auxid, channel=self.wid2.channel,
            tmin=self.wid2.tmin, tmax=self.tmax, deltat=1.0/self.wid2.samprate, ydata=self.data)

class ErrorLog:
    def __init__(self, message):
        self.message = message


class Station:
    
    def __init__(self, network, station, type, lat, lon, coordsys, elevation, ondate, offdate):
        self.network = network
        self.station = station
        self.type = type
        self.lat = lat
        self.lon = lon
        self.coordinate_system = coordsys
        self.elevation = elevation
        self.ondate = ondate
        self.offdate = offdate
        
    def __str__(self):
        # as GSE2.1 format
        return '%-9s %-5s %-4s %9.5f %10.5f %-12s %5.3f %-10s %-10s' % (
            self.network, self.station, self.type, self.lat, self.lon,
            self.coordinate_system, self.elevation, 
            sslashdate(self.ondate), sslashdate(self.offdate))
    
        
class Channel:
    def __init__(self, network, station, channel, auxid, lat, lon, coordsys, elevation, depth, hang, vang, samplerate, instrument, ondate, offdate):
        self.network = network
        self.station = station
        self.channel = channel
        self.auxid = auxid
        self.lat = lat
        self.lon = lon
        self.coordinate_system = coordsys
        self.elevation = elevation
        self.depth = depth
        self.hang = hang
        self.vang = vang
        self.samplerate = samplerate
        self.instrument = instrument
        self.ondate = ondate
        self.offdate = offdate
        
    def __str__(self):
        # as GSE2.1 format
        return '%-9s %-5s %-3s %-4s %9.5f %10.5f %-12s %5.3f %5.3f %6.1f %5.1f %11.6f %-6s %-10s %-10s' % (
            self.network, self.station, self.channel, self.auxid, self.lat, self.lon,
            self.coordinate_system, self.elevation, self.depth, 
            self.hang, self.vang, self.samplerate, self.instrument,
            sslashdate(self.ondate), sslashdate(self.offdate))

class DataSection:
    def __init__(self):
        self.version = None
        self.data_type = None
        self.data = []
        
    def interprete(self, load_data=True):
        mapping = { 'error_log': self.interprete_error_log,
                    'waveform': self.interprete_waveform,
                    'station': self.interprete_station,
                    'channel': self.interprete_channel }
                    
        if self.data_type in mapping:
            for content in mapping[self.data_type](load_data=load_data):
                yield content
        else:
            logger.warn('Skipping unimplemented GSE data type "%s"' % self.data_type)
            
    def interprete_error_log(self, load_data=True):
        message = '\n'.join(self.data+[''])
        yield ErrorLog(message)
        
        
    def interprete_station(self, load_data=True):
        if self.version not in ('GSE2.0', 'GSE2.1'):
            logger.error('Can not interprete GSE station information of version %s' % self.version)
            return
        
        if len(self.data) < 1:
            raise BadGSESection('Need at least header line for station section')
        header = self.data[0].strip()
        if self.version == 'GSE2.0' and  header.lower().split() != 'Sta Type Latitude Longitude Elev On Date Off Date'.lower().split():
            logger.warn('GSE station section header line does not match what is expected for GSE version 2.0')
                        
        if self.version == 'GSE2.1' and header.lower().split() != 'Net Sta Type Latitude Longitude Coord Sys Elev On Date Off Date'.lower().split():
           logger.warn('GSE station section header line does not match what is expected for GSE version 2.1')

        for line in self.data[1:]:
            if self.version == 'GSE2.1':
                net, sta, typ, lat, lon, coordsys, elev, ondate, offdate = unpack_fixed('a9,x1,a5,x1,a4,x1,f9,x1,f10,x1,a12,x1,f5,x1,a10,x1,a10', line)
            elif self.version == 'GSE2.0':
                sta, typ, lat, lon, elev, ondate, offdate = unpack_fixed('a5,x1,a4,x1,f9,x1,f10,x1,f7,x1,a10,x1,a10', line)
                net, coordsys = '', ''
                
            ondate = slashdate(ondate)
            offdate = slashdate(offdate)
                
            yield Station(net, sta, typ, lat, lon, coordsys, elev, ondate, offdate)
            
            
    def interprete_channel(self, load_data=True):
        if self.version not in ('GSE2.0', 'GSE2.1'):
            logger.error('Can not interprete GSE channel information of version %s' % self.version)
            return
        
        if len(self.data) < 1:
             raise BadGSESection('Need at least header line for channel section')
        
        header = self.data[0].strip()
        if self.version == 'GSE2.0' and  header.lower().split() != 'Sta Chan Aux Latitude Longitude Elev Depth Hang Vang Sample_Rate Inst On Date Off Date'.lower().split():
            logger.warn('GSE channel section header line does not match what is expected for GSE version 2.0')
                        
        if self.version == 'GSE2.1' and header.lower().split() != 'Net Sta Chan Aux Latitude Longitude Coord Sys Elev Depth Hang Vang Sample Rate Inst On Date Off Date'.lower().split():
           logger.warn('GSE channel section header line does not match what is expected for GSE version 2.1')
            
        for line in self.data[1:]:
            if self.version == 'GSE2.1':
                net, sta, cha, auxid, lat, lon, coordsys, elev, depth, hang, vang, samprate, instype, ondate, offdate = \
                    unpack_fixed('a9,x1,a5,x1,a3,x1,a4,x1,f9,x1,f10,x1,a12,x1,f5,x1,f5,x1,f6,x1,f5,x1,f11,x1,a6,x1,a10,x1,a10', line)
            elif self.version == 'GSE2.0':
                
                sta, cha, auxid, lat, lon, elev, depth, hang, vang, samprate, instype, ondate, offdate  = \
                    unpack_fixed('a5,x1,a3,x1,a4,x1,f9,x1,f10,x1,f7,x1,f6,x1,f6,x1,f5,x1,f11,x1,a7,x1,a10,x1,a10', line)
                net, coordsys = '', ''
                
            ondate = slashdate(ondate)
            offdate = slashdate(offdate)
            yield Channel(net, sta, cha, auxid, lat, lon, coordsys, elev, depth, hang, vang, samprate, instype, ondate, offdate)
            
    def interprete_waveform(self, load_data=True):
        rawdata_l = []
        at = 0
        wid2, dat2, chk2, sta2 = None, None, None, None
        def reset():
            wid2 = None
            dat2 = None
            chk2 = None
            sta2 = None
            
            
        reset()
        for line in self.data:
            if at in (0,2):
                if line.startswith('WID2'):
                    if wid2: 
                        yield Waveform(wid2, sta2, chk2, dat2)
                        reset()
                    wid2 = Anon()

                    assert line[24:28].startswith('.')
                    
                    
                    wid2.tmin = ( calendar.timegm( time.strptime( 
                            line[5:15]+ ' ' + line[16:24].replace(' ', '0'), 
                            '%Y/%m/%d %H:%M:%S') )
                        + float(line[24:28]))
                        
                    strtmin, wid2.station, wid2.channel, wid2.auxid, wid2.sub_format, \
                        wid2.samps, wid2.samprate, wid2.calib, wid2.calper, \
                        wid2.instype, wid2.hang, wid2.vang = unpack_fixed( \
                        'x5,a23,x1,a5,x1,a3,x1,a4,x1,a3,x1,i8,x1,f11,x1,f10,x1,f7,x1,a6,x1,f5,x1,f4',
                        line)
                    
                    at = 1
                    continue
                    
            if at == 1:
                if line.startswith('STA2'):
                    sta2 = Anon()
                    sta2.network, sta2.lat, sta2.lon, sta2.coordsys, sta2.elev, sta2.depth = unpack_fixed(
                        'x5,a9,x1,f9,x1,f10,x11,a12,x1,f5,x1,f5?', line)
                    
                if line.startswith('EID2'):
                    logger.warn('Cannot handle GSE2 EID2 blocks')
                
                if line.startswith('BEA2'):
                    logger.warn('Cannot handle GSE2 BEA2 blocks')
                
                if line.startswith('DLY2'):
                    logger.warn('Cannot handle GSE2 DLY2 blocks')
                    
                if line.startswith('OUT2'):
                    logger.warn('Cannot handle GSE2 OUT2 blocks')
                    
                
                if line.strip() == 'DAT2':
                    dat2 = Anon()
                    if load_data:
                        dat2.rawdata = []
                    else:
                        dat2.rawdata = None
                    at = 2
                    continue
               
                
            if at == 2:
                if line.startswith('CHK2'):
                    chk2 = Anon()
                    toks = line.split()
                    assert len(toks) == 2
                    chk2.checksum = int(toks[1])
                    at = 0
                    continue
                else:
                    if load_data:
                        dat2.rawdata.append(line)
                    
        if wid2:
            yield Waveform(wid2, sta2, chk2, dat2)
            reset()
                    
    
def readgse(fn, load_data=True):
    
    f = open(fn, 'r')
    
    at = 0
    d = None
    gse = None
    
    for fullline in f:
        line = fullline.rstrip()
        toks = line.split()
        toks1 = line.split(None,1)
        if at in (1,2):
            if isd(line, toks, 'STOP'):
                if gse and d:
                    for content in d.interprete(load_data=load_data):
                        gse.add(content)
                        
                yield gse
                gse = None
                d = None
                
                at = 0
                continue
        
        if at == 0:
            if isd(line, toks, 'BEGIN', 2):
                gse = GSE()
                gse.version = toks[1]
                
                at = 1
                continue
            if line.lstrip().startswith('WID2'):
                gse = GSE()
                gse.version = 'GSE2.1'
                
                d = DataSection()
                d.data_type = 'waveform'
                d.version = 'GSE2.1'

                at = 2
        
        if at == 1:
            if isd(line, toks, 'MSG_TYPE', 2):
                gse.msg_type = toks[1]
            
            if isd(line, toks1, 'MSG_ID', 2):
                gse.msg_id = toks1[1]
            
            if isd(line, toks1, 'REF_ID', 2):
                gse.ref_id = toks1[1]
                        
        if at in (0,1,2):
            if at == 0:
                logger.warn('GSE data has no BEGIN section')
                gse = GSE()
                if len(toks) == 3:
                    gse.version = toks[2]
            
            if isd(line, toks, 'DATA_TYPE', (2,3)):
                if d:
                    for content in d.interprete(load_data=load_data):
                        gse.add( content )
                d = DataSection()
                d.data_type = toks[1].lower()
                if len(toks) == 3:
                    d.version = toks[2]

                at = 2
                continue
                
        if at == 2:
            if line.strip() == '':
                at = 1
                continue
            
            d.data.append(line)
            
    f.close()
    
    if gse:
        if d:
            for content in d.interprete(load_data=load_data):
                gse.add( content )
                
        yield gse
        

def iload(filename, load_data=True):
    for gse in readgse(filename, load_data=load_data):
        for wv in gse.waveforms:
            yield wv.trace()


def detect(first512):
    lines = first512.lstrip().splitlines()
    if len(lines) >= 2:
        if lines[0].startswith('WID2 '):
            return True

        if lines[0].startswith('BEGIN GSE2'):
            return True

    return False


if __name__ == '__main__':
    all_traces = []
    for fn in sys.argv[1:]:
        if detect(open(fn).read(512)):
            all_traces.extend(iload(fn))

    trace.snuffle(all_traces)




