
import util, model
import sys, re, calendar, time, logging
from pyrocko import gse_ext

unpack_fixed = util.unpack_fixed

logger = logging.getLogger('pyrocko.gse')

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

>>>>>>> f6c3aff167bd71a614a59c13f6bb445f8988d919

def isd(line, toks, name, nargs=None):
    if not line.startswith(name):
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
        for attrib in 'tmin station channel auxid sub_format samps samprate calib calper instype hang vang'.split():
            setattr(self, attrib, kwargs[attrib])
      
        assert self.sub_format in 'INT CM6 CM8 AUT AU6 AU8'.split()
        
        print ext_gse.decode_m6( dat2.rawdata )
        
    def __str__(self):
        return ' '.join([self.station, self.channel, self.auxid, self.sub_format, util.gmctime(self.tmin)])


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
        
    def interprete(self):
        mapping = { 'error_log': self.interprete_error_log,
                    'waveform': self.interprete_waveform,
                    'station': self.interprete_station,
                    'channel': self.interprete_channel }
                    
        if self.data_type in mapping:
            for content in mapping[self.data_type]():
                yield content
        else:
            logger.warn('Skipping unimplemented GSE data type "%s"' % self.data_type)
            
    def interprete_error_log(self):
        message = '\n'.join(self.data+[''])
        yield ErrorLog(message)
        
        
    def interprete_station(self):
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
            
            
    def interprete_channel(self):
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
            
    def interprete_waveform(self):
        rawdata_l = []
        at = 0
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
                        yield wid2(wid2, chk2, dat2)
                        reset()
                    wid2 = Anon()

                    assert line[24:28].startswith('.')
                    wid2.tmin = ( calendar.timegm( time.strptime( 
                            line[5:15]+ ' ' + line[16:24], 
                            '%Y/%m/%d %H:%M:%S') )
                        + float(line[24:28]))
                    wid2.station = line[29:34].strip()
                    wid2.channel = line[35:38].strip()
                    wid2.auxid = line[39:43].strip()
                    wid2.sub_format = line[44:47]
                    wid2.samps = int(line[48:56])
                    wid2.samprate = float(line[57:68])
                    wid2.calib = float(line[69:79])
                    wid2.calper = float(line[80:87])
                    wid2.instype = line[88:94].strip()
                    wid2.hang = float(line[95:100])
                    wid2.vang = float(line[101:105])
                    at = 1
                    continue
                    
            if at == 1:
                if line.startswith('STA2'):
                    sta2 = Anon()
                    sta2.network = line[5:14].strip()
                    sta2.lat = float(line[15:34])
                    sta2.lon = float(line[35:45])
                    sta2.coordsys = line[46:58].strip()
                    sta2.elev = float(line[59:64])
                    sta2.edepth = float(line[65:70])
                    
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
                    dat2.rawdata = []
                    at = 2
                    continue
               
                
            if at == 2:
                if line.startswith('CHK2'):
                    toks = line.split()
                    assert len(toks) == 2
                    chk2.checksum = int(toks[1])
                    at = 0
                    continue
                else:
                    dat2.rawdata.append(line)
                    
        if waveform:
            yield Waveform(**waveform.__dict__)
            reset()
                    
    
def readgse(fn):
    
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
                    for content in d.interprete():
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
                    for content in d.interprete():
                        gse.add( content )
                d = DataSection()
                d.data_type = toks[1]
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
            for content in d.interprete():
                gse.add( content )
                
        yield gse
        
