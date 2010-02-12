'''SAC IO library for Python'''

import struct, sys, logging, math, time
from calendar import timegm
from time import gmtime
import numpy as num

class SacError(Exception):
    pass

class SacFile:
    nbytes_header = 632
    header_num_format = {'little': '<70f40i', 'big': '>70f40i'}
    
    header_keys = '''
delta depmin depmax scale odelta b e o a internal0 t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 
f resp0 resp1 resp2 resp3 resp4 resp5 resp6 resp7 resp8 resp9 stla stlo stel 
stdp evla evlo evel evdp mag user0 user1 user2 user3 user4 user5 user6 user7 
user8 user9 dist az baz gcarc internal1 internal2 depmen cmpaz cmpinc xminimum 
xmaximum yminimum ymaximum unused0 unused1 unused2 unused3 unused4 unused5 
unused6 nzyear nzjday nzhour nzmin nzsec nzmsec nvhdr norid nevid npts internal3
nwfid nxsize nysize unused7 iftype idep iztype unused8 iinst istreg ievreg
ievtyp iqual isynth imagtyp imagsrc unused9 unused10 unused11 unused12 unused13
unused14 unused15 unused16 leven lpspol lovrok lcalda unused17 kstnm kevnm khole
ko ka kt0 kt1 kt2 kt3 kt4 kt5 kt6 kt7 kt8 kt9 kf kuser0 kuser1 kuser2 kcmpnm 
knetwk kdatrd kinst
'''.split()
    
    header_enum_symbols = '''
itime irlim iamph ixy iunkn idisp ivel iacc ib iday io ia it0 it1 it2 it3 it4 
it5 it6 it7 it8 it9 iradnv itannv iradev itanev inorth ieast ihorza idown iup 
illlbb iwwsn1 iwwsn2 ihglp isro inucl ipren ipostn iquake ipreq ipostq ichem 
iother igood iglch idrop ilowsn irldta ivolts ixyz imb ims iml imw imd imx ineic
ipde iisc ireb iusgs ibrk icaltech illnl ievloc ijsop iuser iunknown iqb iqb1 
iqb2 iqbx iqmt ieq ieq1 ieq2 ime iex inu inc io_ il ir it iu
'''.split()
    
    header_num2name = dict([ (a+1,b) for (a,b) in enumerate(header_enum_symbols)])
    header_name2num = dict([ (b,a+1) for (a,b) in enumerate(header_enum_symbols)])
    header_types = 'f'*70 + 'i'*35 + 'l'*5 + 'k'*23
    undefined_value = {'f':-12345.0, 'i':-12345, 'l':None, 'k': '-12345'}
    ldefaults = {'leven': 1, 'lpspol': 0, 'lovrok': 1, 'lcalda': 1, 'unused17': 0}
    t_lookup = dict(zip(header_keys, header_types))
    u_lookup = dict([ (k, undefined_value[t_lookup[k]]) for k in header_keys ])
    
    def ndatablocks(self):
        '''Get number of data blocks for this file's type.'''
        nblocks = { 'itime':1, 'irlim':2, 'iamph':2, 'ixy':2, 'ixyz':3 }[SacFile.header_num2name[self.iftype]]
        if nblocks == 1 and not self.leven: 
            nblocks = 2 # not sure about this...
        return nblocks

    def val_or_none(self, k,v):
        '''Replace SAC undef flags with None.'''
        if SacFile.u_lookup[k] == v:
            return None
        else:
            return v
    
    def get_ref_time(self):
        '''Get reference time as standard Unix timestamp.'''
        
        if None in (self.nzyear, self.nzjday, self.nzhour, self.nzmin, self.nzsec, self.nzmsec):
            raise SacError('Not all header values for reference time are set.')
        return timegm((self.nzyear, 1, self.nzjday, self.nzhour, self.nzmin, self.nzsec)) + self.nzmsec/1000.
    
    def set_ref_time(self, timestamp):
        '''Set all header values defining reference time based on standard Unix timestamp.'''
        
        secs = math.floor(timestamp)
        msec = int(round((timestamp-secs)*1000.))
        if msec == 1000:
            secs += 1
            msec = 0
        
        t = gmtime(secs)
        self.nzyear, self.nzjday, self.nzhour, self.nzmin, self.nzsec = t[0], t[7], t[3], t[4], t[5]
        self.nzmsec = msec
        
    def val_for_file(self, k,v):
        '''Convert attribute value to the form required when writing it to the SAC file.'''
        
        t = SacFile.t_lookup[k]
        if v is None:
            if t == 'l': return SacFile.ldefaults[k]
            v = SacFile.u_lookup[k]
        if t == 'f': return float(v)
        elif t == 'i': return int(v)
        elif t == 'l':
            if v: return 1
            return 0
        elif t == 'k':
            l = 8
            if k == 'kevnm': l = 16   # only this header val has different length
            return v.ljust(l)[:l]
            
    def __init__(self, *args, **kwargs):
        if args:
            self.read(*args, **kwargs)
        else:
            self.clear()
    
    def clear(self):
        '''Empty file record.'''
        
        for k in SacFile.header_keys:
            self.__dict__[k] = None
        
        # set the required attributes        
        self.nvhdr = 6
        self.iftype = SacFile.header_name2num['itime']
        self.leven = True
        self.delta = 1.0
        self.npts = 0
        self.b = 0.0
        self.e = 0.0
        self.data = [ num.arange(0, dtype=num.float32) ]
        
    def check(self):
        '''Check the required header variables to have reasonable values.'''
    
        if self.iftype not in [SacFile.header_name2num[x] for x in ('itime','irlim','iamph','ixy','ixyz')]:
            raise SacError('Unknown SAC file type: %i.' % self.iftype)
        if self.nvhdr < 1 or 20 < self.nvhdr:
            raise SacError('Unreasonable SAC header version number found.') 
        if self.npts < 0:
            raise SacError('Negative number of samples specified in NPTS header.')
        if self.leven not in (0,1):
            raise SacError('Header value LEVEN must be either 0 or 1.')
        if self.leven and self.delta <= 0.0:
            raise SacError('Header value DELTA should be positive for evenly spaced samples')
        if self.b > self.e:
            raise SacError('Beginning value of independent variable greater than its ending value.') 
        if self.nvhdr != 6:
            logging.warn('This module has only been tested with SAC header version 6.'+
                         'This file has header version %i. It might still work though...' % self.nvhdr)

    def read(self, filename, get_data=True, byte_sex='try'):
        '''Read SAC file.
        
           filename -- Name of SAC file.
           get_data -- If True, the data is read, otherwise only read headers.
           byte_sex -- Endianness: 'try', 'little' or 'big' 
        '''
        nbh = SacFile.nbytes_header
        
        # read in all data
        f = open(filename,'rb')
        if get_data:
            filedata = f.read()
        else:
            filedata = f.read(nbh)
        f.close()
            
        if len(filedata) < nbh:
            raise SacError('File too short to be a SAC file.')

        # possibly try out different endiannesses        
        if byte_sex == 'try':
            sexes = ('little','big')
        else:
            sexes = (byte_sex,)
        
        for isex,sex in enumerate(sexes):
            format = SacFile.header_num_format[sex]
            nbn = struct.calcsize(format)
            hv = list(struct.unpack(format, filedata[:nbn]))
            
            strings = filedata[nbn:nbh]
            hv.append(strings[:8].rstrip())
            hv.append(strings[8:24].rstrip())
            for i in xrange(len(strings[24:])/8):
                hv.append(strings[24+i*8:24+(i+1)*8].rstrip())
            
            self.header_vals = hv
            for k, v in zip(SacFile.header_keys, self.header_vals):
                self.__dict__[k] = self.val_or_none(k,v)
            
            self.data = []
            try:
                self.check()
                break
            except SacError, e:
                if isex == len(sexes)-1:
                    raise e
        
        # possibly get data
        if get_data:
            nblocks = self.ndatablocks()
            nbb = self.npts*4 # word length is always 4 bytes in sac files
            for iblock in range(nblocks):
                if len(filedata) < nbh+(iblock+1)*nbb:
                    raise SacError('File is incomplete.')
                    
                self.data.append(num.fromstring(filedata[nbh+iblock*nbb:nbh+(iblock+1)*nbb], dtype=num.float32))
            
            if len(filedata) > nbh+nblocks*nbb:
                sys.stderr.warn('Unused data at end of file.')
            
            
    def write(self, filename, byte_sex='little'):
        '''Write SAC file.'''
        
        self.check()
        
        # create header data
        format = SacFile.header_num_format[byte_sex]
        nbn = struct.calcsize(format)
        numerical_values = []
        string_values = []
        for k in SacFile.header_keys:
            v = self.__dict__[k]
            vv = self.val_for_file(k,v)
            if SacFile.t_lookup[k] == 'k':
                string_values.append(vv)
            else:
                numerical_values.append(vv)
            
        header_data = struct.pack(format, *numerical_values)
        header_data += ''.join(string_values)
        
        # check that enough data is available
        nblocks = self.ndatablocks()
        if len(self.data) != nblocks:
            raise SacError('Need %i data blocks for file type %s.' % ( nblocks, SacFile.header_num2name[self.iftype] ))
        
        for fdata in self.data:
            if len(fdata) != self.npts:
                raise SacError('Data length (%i) does not match NPTS header value (%i)' % (len(fdata), self.npts))
        
        # dump data to file
        f = open(filename, 'wb')
        f.write(header_data)
        for fdata in self.data:
            f.write(fdata.astype(num.float32).tostring())
        f.close()
        
    def __str__(self):
        str = ''
        for k in SacFile.header_keys:
            v = self.__dict__[k]
            if v is not None:
                str += '%s: %s\n' % (k, v)
                        
        return str
        
if __name__ == "__main__":
    print SacFile(sys.argv[1])

    s = SacFile()
    fn = '/tmp/x.sac'
    secs = timegm((2009,2,19,8,50,0))+0.1
    s.set_ref_time(secs)
    s.write(fn)
    s2 = SacFile(fn)
    assert(s2.nzjday == 50)
    assert(s2.nzmsec == 100)
    
    