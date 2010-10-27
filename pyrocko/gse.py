
import util

import sys, re, calendar, time, logging

logger = logging.get_logger('pyrocko.gse')

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
    
    def add(self, content):
        if isinstance(content, Waveform):
            self.waveforms.append(content)
        if isinstance(content, ErrorLog):
            self.error_logs.append(content)
            
    def __str__(self):
        s = []
        for waveform in self.waveforms:
            s.append(str(waveform))

        s.append('')
        return '\n'.join(s)

class Waveform:
    def __init__(self, wid2, sta2, chk2, dat2):
        for attrib in 'tmin station channel auxid sub_format samps samprate calib calper instype hang vang'.split():
            setattr(self, attrib, kwargs[attrib])
      
        assert self.sub_format in 'INT CM6 CM8 AUT AU6 AU8'.split()

        
    def __str__(self):
        return ' '.join([self.station, self.channel, self.auxid, self.sub_format, util.gmctime(self.tmin)])

        
class ErrorLog:
    def __init__(self, message):
        self.message = message


class DataSection:
    def __init__(self):
        self.version = None
        self.data_type = None
        self.data = []
        
    def interprete(self):
        print self.data_type
        mapping = { 'error_log': self.interprete_error_log,
                    'waveform': self.interprete_waveform }
                    
        if self.data_type in mapping:
            for content in mapping[self.data_type]():
                yield content
    
    def interprete_error_log(self):
        message = '\n'.join(self.data+[''])
        yield ErrorLog(message)
        
        
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
                        
        if at in (1,2):
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
            d.data.append(line)
            
    f.close()
    if gse:
        yield gse
        


for gse in readgse(sys.argv[1]):
    print gse
    

