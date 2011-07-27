
import util, evalresp
import time, math, copy, logging
import numpy as num
from util import reuse
from scipy import signal
from pyrocko import model
from nano import asnano, Nano

logger = logging.getLogger('pyrocko.trace')


def minmax(traces, key=lambda tr: (tr.network, tr.station, tr.location, tr.channel), mode='minmax'):
    
    '''Get data range given traces grouped by selected pattern.
    
    A dict with the combined data ranges is returned. By default, the keys of
    the output dict are tuples formed from the selected keys out of network,
    station, location, and channel, in that particular order.
    '''
    
    ranges = {}
    for trace in traces:
        if isinstance(mode, str) and mode == 'minmax':
            mi, ma = trace.ydata.min(), trace.ydata.max()
        else:
            mean = trace.ydata.mean()
            std = trace.ydata.std()
            mi, ma = mean-std*mode, mean+std*mode
            
        k = key(trace)
        if k not in ranges:
            ranges[k] = mi, ma
        else:
            tmi, tma = ranges[k]
            ranges[k] = min(tmi,mi), max(tma,ma)
    
    return ranges
        
def minmaxtime(traces, key=lambda tr: (tr.network, tr.station, tr.location, tr.channel)):
    
    '''Get time range given traces grouped by selected pattern.
    
    A dict with the combined time ranges is returned. By default, the keys of
    the output dict are tuples formed from the selected keys out of network,
    station, location, and channel, in that particular order.
    '''
    
    ranges = {}
    for trace in traces:
        mi, ma = trace.tmin, trace.tmax
        k = key(trace)
        if k not in ranges:
            ranges[k] = mi, ma
        else:
            tmi, tma = ranges[k]
            ranges[k] = min(tmi,mi), max(tma,ma)
    
    return ranges
    
def degapper(in_traces, maxgap=5, fillmethod='interpolate', deoverlap='use_second'):
    
    '''Try to connect traces and remove gaps.
    
    This method will combine adjacent traces, which match in their network, 
    station, location and channel attributes. Overlapping parts are handled
    according to the `deoverlap` argument.
    
    Arguments:
    
       in_traces:   input traces, must be sorted by their full_id attribute.
       maxgap:      maximum number of samples to interpolate.
       fillmethod:  what to put into the gaps: 'interpolate' or 'zeros'.
       deoverlap:   how to handle overlaps: 'use_second' to use data from 
                    second trace (default), 'use_first' to use data from first
                    trace, 'crossfade_cos' to crossfade with cosine taper 
       
    '''
    
    out_traces = []
    if not in_traces: return out_traces
    out_traces.append(in_traces.pop(0))
    while in_traces:
        
        a = out_traces[-1]
        b = in_traces.pop(0)

        if (a.nslc_id == b.nslc_id and a.deltat == b.deltat and 
            len(a.ydata) >= 1 and len(b.ydata) >= 1 and a.ydata.dtype == b.ydata.dtype):
            
            dist = (b.tmin-(a.tmin+(len(a.ydata)-1)*a.deltat))/a.deltat
            idist = int(round(dist))
            if abs(dist - idist) > 0.05 and idist <= maxgap:
                pass #logger.warn('Cannot degap traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
            else:
                if 1 < idist <= maxgap:
                    if fillmethod == 'interpolate':
                        filler = a.ydata[-1] + (((1.+num.arange(idist-1,dtype=num.float))/idist)*(b.ydata[0]-a.ydata[-1])).astype(a.ydata.dtype)
                    elif fillmethod == 'zeros':
                        filler = num.zeros(idist-1,dtype=a.ydist.dtype)
                    a.ydata = num.concatenate((a.ydata,filler,b.ydata))
                    a.tmax = b.tmax
                    if a.mtime and b.mtime:
                        a.mtime = max(a.mtime, b.mtime)
                    continue

                elif idist == 1:
                    a.ydata = num.concatenate((a.ydata,b.ydata))
                    a.tmax = b.tmax
                    if a.mtime and b.mtime:
                        a.mtime = max(a.mtime, b.mtime)
                    continue
                    
                elif idist <= 0:
                    if b.tmax > a.tmax:
                        na = a.ydata.size
                        n = -idist+1
                        if deoverlap == 'use_second':
                            a.ydata = num.concatenate((a.ydata[:-n], b.ydata))
                        elif deoverlap in ('use_first', 'crossfade_cos'):
                            a.ydata = num.concatenate((a.ydata, b.ydata[n:]))
                        else:
                            assert False, 'unknown deoverlap method'

                        if deoverlap == 'crossfade_cos':
                            n = -idist+1
                            taper = 0.5-0.5*num.cos((1.+num.arange(n))/(1.+n)*num.pi)
                            a.ydata[na-n:na] *= 1.-taper
                            a.ydata[na-n:na] += b.ydata[:n] * taper

                        a.tmax = b.tmax
                        if a.mtime and b.mtime:
                            a.mtime = max(a.mtime, b.mtime)
                        continue
                    else:
                        # make short second trace vanish
                        continue
                    
        if len(b.ydata) >= 1:
            out_traces.append(b)
            
    for tr in out_traces:
        tr.update_ids()
    
    return out_traces

def rotate(traces, azimuth, in_channels, out_channels):
    '''Rotate corresponding traces
    
    In:
       traces -- not rotated traces
       azimuth -- difference of the azimuths of the component directions
                     (azimuth of out_channels[0]) - (azimuth of in_channels[0])
       in_channels -- names of the input channels (e.g. 'N', 'E')
       out_channels -- names of the output channels (e.g. 'R', 'T')
       
    '''
    
    phi = azimuth/180.*math.pi
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    rotated = []
    in_channels = tuple(channels_to_names(in_channels))
    out_channels = tuple(channels_to_names(out_channels))
    for a in traces:
        for b in traces:
            if ( (a.channel, b.channel) == in_channels and
                 a.nslc_id[:3] == b.nslc_id[:3] and
                 abs(a.deltat-b.deltat) < a.deltat*0.001 ):
                tmin = max(a.tmin, b.tmin)
                tmax = min(a.tmax, b.tmax)
                
                if tmin < tmax:
                    ac = a.chop(tmin, tmax, inplace=False, include_last=True)
                    bc = b.chop(tmin, tmax, inplace=False, include_last=True)
                    if abs(ac.tmin - bc.tmin) > ac.deltat*0.01:
                        logger.warn('Cannot rotate traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
                        continue
                    
                    acydata =  ac.get_ydata()*cphi+bc.get_ydata()*sphi
                    bcydata = -ac.get_ydata()*sphi+bc.get_ydata()*cphi
                    ac.set_ydata(acydata)
                    bc.set_ydata(bcydata)

                    ac.set_codes(channel=out_channels[0])
                    bc.set_codes(channel=out_channels[1])
                    rotated.append(ac)
                    rotated.append(bc)
                    
    return rotated


def decompose(a):
    '''Decompose matrix into independent submatrices.'''
    
    def depends(iout,a):
        row = a[iout,:]
        return set(num.arange(row.size).compress(row != 0.0))
    
    def provides(iin,a):
        col = a[:,iin]
        return set(num.arange(col.size).compress(col != 0.0))
    
    a = num.asarray(a)
    outs = set(range(a.shape[0]))
    systems = []
    while outs:
        iout = outs.pop()
        
        gout = set()
        for iin in depends(iout,a):
            gout.update(provides(iin,a))
        
        if not gout: continue
        
        gin = set()
        for iout2 in gout:
            gin.update(depends(iout2,a))
        
        if not gin: continue
                
        for iout2 in gout:
            if iout2 in outs:
                outs.remove(iout2)

        gin = list(gin)
        gin.sort()
        gout = list(gout)
        gout.sort()
        
        systems.append((gin, gout, a[gout,:][:,gin]))
    
    return systems

def channels_to_names(channels):
    names = []
    for ch in channels:
        if isinstance(ch, model.Channel):
            names.append(ch.name)
        else:
            names.append(ch)
    return names

def project(traces, matrix, in_channels, out_channels):
    
    # try to apply transformation to subsets of the channels if this is 
    # possible, such that if for example a vertical component is missing,
    # the horizontal components can still be rotated.
    
    in_channels = tuple( channels_to_names(in_channels) )
    out_channels = tuple( channels_to_names(out_channels) )
    systems = decompose(matrix)
    
    # fallback to full matrix if some are not quadratic
    for iins, iouts, submatrix in systems:
        if submatrix.shape[0] != submatrix.shape[1]:
            return project3(traces, matrix, in_channels, out_channels)
    
    projected = []
    for iins, iouts ,submatrix in systems:
        in_cha = tuple( [ in_channels[iin] for iin in iins ] )
        out_cha = tuple( [ out_channels[iout] for iout in iouts ] )
        if submatrix.shape[0] == 1:
            projected.extend( project1(traces, submatrix, in_cha, out_cha) )
        elif submatrix.shape[1] == 2:
            projected.extend( project2(traces, submatrix, in_cha, out_cha) )
        else:
            projected.extend( project3(traces, submatrix, in_cha, out_cha) )
    
   
    return projected

def project_dependencies(matrix, in_channels, out_channels):
    
    # figure out what dependencies project() would produce
    
    in_channels = tuple( channels_to_names(in_channels) )
    out_channels = tuple( channels_to_names(out_channels) )
    systems = decompose(matrix)
    
    subpro = []
    for iins, iouts, submatrix in systems:
        if submatrix.shape[0] != submatrix.shape[1]:
            subpro.append((matrix, in_channels, out_channels))
    
    if not subpro:
        for iins, iouts ,submatrix in systems:
            in_cha = tuple( [ in_channels[iin] for iin in iins ] )
            out_cha = tuple( [ out_channels[iout] for iout in iouts ] )
            subpro.append((submatrix, in_cha, out_cha))
            
    deps = {}
    for mat, in_cha, out_cha in subpro:
        for oc in out_cha:
            if oc not in deps:
                deps[oc] = []
            
            for ic in in_cha:
                deps[oc].append(ic)
    
    return deps
        
def project1(traces, matrix, in_channels, out_channels):
    assert len(in_channels) == 1
    assert len(out_channels) == 1
    assert matrix.shape == (1,1)
    
    projected = []
    for a in traces:
        if not (a.channel,) == in_channels: 
            continue
        
        ac = a.copy()
        ac.set_ydata(matrix[0,0]*a.get_ydata())
        ac.set_codes(channel=out_channels[0])
        projected.append(ac)
        
    return projected

def project2(traces, matrix, in_channels, out_channels):
    assert len(in_channels) == 2
    assert len(out_channels) == 2
    assert matrix.shape == (2,2)
    projected = []
    for a in traces:
        for b in traces:
            if not ( (a.channel, b.channel ) == in_channels and
                    a.nslc_id[:3] == b.nslc_id[:3] and
                    abs(a.deltat-b.deltat) < a.deltat*0.001 ):
                continue
                    
            tmin = max(a.tmin, b.tmin)
            tmax = min(a.tmax, b.tmax)
            
            if tmin >= tmax:
                continue
        
            ac = a.chop(tmin, tmax, inplace=False, include_last=True)
            bc = b.chop(tmin, tmax, inplace=False, include_last=True)
            if abs(ac.tmin - bc.tmin) > ac.deltat*0.01:
                logger.warn('Cannot project traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
                continue
                
            acydata = num.dot( matrix[0], (ac.get_ydata(),bc.get_ydata()))
            bcydata = num.dot( matrix[1], (ac.get_ydata(),bc.get_ydata()))
            
            ac.set_ydata(acydata)
            bc.set_ydata(bcydata)

            ac.set_codes(channel=out_channels[0])
            bc.set_codes(channel=out_channels[1])
            
            projected.append(ac)
            projected.append(bc)
    
    return projected
            
def project3(traces, matrix, in_channels, out_channels):
    assert len(in_channels) == 3
    assert len(out_channels) == 3
    assert matrix.shape == (3,3)
    projected = []
    for a in traces:
        for b in traces:
            for c in traces:
                if not ( (a.channel, b.channel, c.channel) == in_channels and
                     a.nslc_id[:3] == b.nslc_id[:3] and
                     b.nslc_id[:3] == c.nslc_id[:3] and
                     abs(a.deltat-b.deltat) < a.deltat*0.001 and
                     abs(b.deltat-c.deltat) < b.deltat*0.001 ):
                    continue
                     
                tmin = max(a.tmin, b.tmin, c.tmin)
                tmax = min(a.tmax, b.tmax, c.tmax)
                    
                if tmin >= tmax:
                    continue
                
                ac = a.chop(tmin, tmax, inplace=False, include_last=True)
                bc = b.chop(tmin, tmax, inplace=False, include_last=True)
                cc = c.chop(tmin, tmax, inplace=False, include_last=True)
                if (abs(ac.tmin - bc.tmin) > ac.deltat*0.01 or
                    abs(bc.tmin - cc.tmin) > bc.deltat*0.01):
                    logger.warn('Cannot project traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
                    continue
                    
                acydata = num.dot( matrix[0],
                    (ac.get_ydata(),bc.get_ydata(),cc.get_ydata()))
                bcydata = num.dot( matrix[1],
                    (ac.get_ydata(),bc.get_ydata(),cc.get_ydata()))
                ccydata = num.dot( matrix[2],
                    (ac.get_ydata(),bc.get_ydata(),cc.get_ydata()))
                
                ac.set_ydata(acydata)
                bc.set_ydata(bcydata)
                cc.set_ydata(ccydata)

                ac.set_codes(channel=out_channels[0])
                bc.set_codes(channel=out_channels[1])
                cc.set_codes(channel=out_channels[2])
                
                projected.append(ac)
                projected.append(bc)
                projected.append(cc)
     
    return projected

def moving_avg(x,n):
    n = int(n)
    cx = x.cumsum()
    nn = len(x)
    y = num.zeros(nn)
    y[n/2:n/2+(nn-n)] = (cx[n:]-cx[:-n])/n
    y[:n/2] = y[n/2]
    y[n/2+(nn-n):] = y[n/2+(nn-n)-1]
    return y

def nextpow2(i):
    return 2**int(math.ceil(math.log(i)/math.log(2.)))
    
def snapper(nmax, delta, snapfun=math.ceil):
    def snap(x):
        return max(0,min(snapfun(x/delta),nmax))
    return snap

def costaper(a,b,c,d, nfreqs, deltaf):
    hi = snapper(nfreqs, deltaf)
    tap = num.zeros(nfreqs)
    tap[hi(a):hi(b)] = 0.5 - 0.5*num.cos((deltaf*num.arange(hi(a),hi(b))-a)/(b-a)*num.pi)
    tap[hi(b):hi(c)] = 1.
    tap[hi(c):hi(d)] = 0.5 + 0.5*num.cos((deltaf*num.arange(hi(c),hi(d))-c)/(d-c)*num.pi)
    
    return tap

def t2ind(t,tdelta, snap=round):
    return int(snap(t/tdelta))


class TraceTooShort(Exception):
    pass

class FrequencyResponse(object):
    '''Evaluates frequency response at given frequencies.'''
    
    def evaluate(self, freqs):
        coefs = num.ones(freqs.size, dtype=num.complex)
        return coefs
   
class InverseEvalresp(FrequencyResponse):
    '''Calls evalresp and generates values of the inverse instrument response for 
       deconvolution of instrument response.'''
    
    def __init__(self, respfile, trace, target='dis'):
        self.respfile = respfile
        self.nslc_id = trace.nslc_id
        self.instant = trace.tmin
        self.target = target
        
    def evaluate(self, freqs):
        network, station, location, channel = self.nslc_id
        x = evalresp.evalresp(sta_list=station,
                          cha_list=channel,
                          net_code=network,
                          locid=location,
                          instant=self.instant,
                          freqs=freqs,
                          units=self.target.upper(),
                          file=self.respfile,
                          rtype='CS')
        
        transfer = x[0][4]
        return 1./transfer

class PoleZeroResponse(FrequencyResponse):
    
    def __init__(self, zeros, poles, constant):
        self.zeros = zeros
        self.poles = poles
        self.constant = constant
        
    def evaluate(self, freqs):
        jomeg = 1.0j* 2.*num.pi*freqs
        
        a = num.ones(freqs.size, dtype=num.complex)*self.constant
        for z in self.zeros:
            a *= jomeg-z
        for p in self.poles:
            a /= jomeg-p
        
        return a
        
class SampledResponse(FrequencyResponse):
    
    def __init__(self, freqs, vals, left=None, right=None):
        self.freqs = freqs.copy()
        self.vals = vals.copy()
        self.left = left
        self.right = right
        
    def evaluate(self, freqs):
        ereal = num.interp(freqs, self.freqs, num.real(self.vals), left=self.left, right=self.right)
        eimag = num.interp(freqs, self.freqs, num.imag(self.vals), left=self.left, right=self.right)
        transfer = ereal + 1.0j*eimag
        return transfer
    
    def inverse(self):
        return SampledResponse(self.freqs, 1./self.vals, left=self.left, right=self.right)
    
    def frequencies(self):
        return self.freqs
    
    def values(self):
        return self.vals
    
class IntegrationResponse(FrequencyResponse):
    def __init__(self, gain=1.0):
        self._gain = gain
        
    def evaluate(self, freqs):
        return self._gain/(1.0j * 2. * num.pi*freqs)

class DifferentiationResponse(FrequencyResponse):
    def __init__(self, gain=1.0):
        self._gain = gain
        
    def evaluate(self, freqs):
        return self._gain * 1.0j * 2. * num.pi * freqs

class AnalogFilterResponse(FrequencyResponse):
    def __init__(self, b,a):
        self._b = b
        self._a = a
    
    def evaluate(self, freqs):
        return signal.freqs(self._b, self._a, freqs)[1]

class NoData(Exception):
    pass

class AboveNyquist(Exception):
    pass

class Trace(object):
    
    cached_frequencies = {}
        
    def __init__(self, network='', station='STA', location='', channel='', 
                 tmin=0., tmax=None, deltat=1., ydata=None, mtime=None, meta=None):
    
        '''Create new trace object
           
        In:
            network -- network code
            station -- station code
            location -- location code
            channel -- channel code
            tmin -- system time of first sample in [s]
            tmax -- system time of last sample in [s] (if None it is computed from length)
            deltat -- sampling interval in [s]
            ydata -- 1D numpy array with data samples
            mtime -- opitional modification time 
            meta -- additional meta information (not used by pyrocko)
        '''
    
        self._growbuffer = None

        if deltat < 0.001:
            tmin = asnano(tmin)
            if tmax is not None:
                tmax = asnano(tmax)

    
        if mtime is None:
            mtime = time.time()
        
        self.network, self.station, self.location, self.channel = [reuse(x) for x in (network,station,location,channel)]
        
        self.tmin = tmin
        self.deltat = deltat
        
        if tmax is None:
            if ydata is not None:
                self.tmax = self.tmin + (ydata.size-1)*self.deltat
            else:
                raise Exception('fixme: trace must be created with tmax or ydata')
        else:
            self.tmax = tmax
        self.meta = meta
        self.ydata = ydata
        self.mtime = mtime
        self.update_ids()
        
    def __str__(self):
        fmt = min(9, max(0, -int(math.floor(math.log10(self.deltat)))))
        
        s = 'Trace (%s, %s, %s, %s)\n' % self.nslc_id
        s += '  timerange: %s - %s\n' % (util.time_to_str(self.tmin, format=fmt), util.time_to_str(self.tmax, format=fmt))
        s += '  delta t: %g\n' % self.deltat
        if self.meta:
            for k in sorted(self.meta.keys()):
                s += '  %s: %s\n' % (k,self.meta[k])
        return s
        
    def name(self):
        s = '%s.%s.%s.%s, %s, %s' % (self.nslc_id + (util.gmctime(self.tmin), util.gmctime(self.tmax)))
        return s
        
    def __eq__(self, other):
        return (self.network == other.network and
                self.station == other.station and
                self.location == other.location and
                self.channel == other.channel and
                self.deltat == other.deltat and
                abs(self.tmin-other.tmin) < self.deltat*0.001 and
                abs(self.tmax-other.tmax) < self.deltat*0.001 and
                num.all(self.ydata == other.ydata))
    
    def __call__(self, t, clip=False, snap=round):
        it = int(snap((t-self.tmin)/self.deltat))
        if clip:
            it = max(0, min(it, self.ydata.size-1))
        else:
            if it < 0 or self.ydata.size <= it:
                raise IndexError()
                       
        return self.tmin+it*self.deltat, self.ydata[it]
    
    def interpolate(self, t, clip=False):
        t0, y0 = self(t, clip=clip, snap=math.floor)
        t1, y1 = self(t, clip=clip, snap=math.ceil)
        if t0 == t1:
            return y0
        else:
            return y0+(t-t0)/(t1-t0)*(y1-y0)
        
    def index_clip(self, i):
        return min(max(0,i), self.ydata.size)

        
    def add(self, other, interpolate=True):
        
        if interpolate:
            other_xdata = other.get_xdata()
            xdata = self.get_xdata()
            xmin, xmax = other_xdata[0], other_xdata[-1]
            self.ydata += num.interp(xdata, other_xdata, other.ydata, left=0., right=0.)
        else:
            assert self.deltat == other.deltat
            ibeg1 = int(round((other.tmin-self.tmin)/self.deltat))
            ibeg2 = int(round((self.tmin-other.tmin)/self.deltat))
            iend1 = int(round((other.tmax-self.tmin)/self.deltat))+1
            iend2 = int(round((self.tmax-other.tmin)/self.deltat))+1
            
            ibeg1 = self.index_clip(ibeg1)
            iend1 = self.index_clip(iend1)
            ibeg2 = self.index_clip(ibeg2)
            iend2 = self.index_clip(iend2)
            
            self.ydata[ibeg1:iend1] += other.ydata[ibeg2:iend2]
                                    
    def set_codes(self, network=None, station=None, location=None, channel=None):
        if network is not None:
            self.network = network
        if station is not None:
            self.station = station
        if location is not None:
            self.location = location
        if channel is not None:
            self.channel = channel
        
        self.update_ids()
        
    def overlaps(self, tmin,tmax):
        return not (tmax < self.tmin or self.tmax < tmin)
           
    def is_relevant(self, tmin, tmax, selector=None):
        return  not (tmax <= self.tmin or self.tmax < tmin) and (selector is None or selector(self))

    def update_ids(self):
        self.full_id = (self.network,self.station,self.location,self.channel,self.tmin)
        self.nslc_id = reuse((self.network,self.station,self.location,self.channel))

    def set_mtime(self, mtime):
        self.mtime = mtime

    def get_xdata(self):
        if self.ydata is None: raise NoData()
        return self.tmin + num.arange(len(self.ydata), dtype=num.float64) * self.deltat

    def get_ydata(self):
        if self.ydata is None: raise NoData()
        return self.ydata
        
    def set_ydata(self, new_ydata):
        self.drop_growbuffer()
        self.ydata = new_ydata
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        
    def drop_data(self):
        self.drop_growbuffer()
        self.ydata = None
   
    def drop_growbuffer(self):
        self._growbuffer = None

    def copy(self, data=True):
        tracecopy = copy.copy(self)
        self.drop_growbuffer()
        if data:
            tracecopy.ydata = self.ydata.copy()
        tracecopy.meta = copy.deepcopy(self.meta)
        return tracecopy
    
    def crop_zeros(self):
        '''Remove zeros at beginning and end.'''
        
        indices = num.where(self.ydata != 0.0)[0]
        if indices.size == 0:
            raise NoData()
        
        ibeg = indices[0]
        iend = indices[-1]+1
        if ibeg == 0 and iend == self.ydata.size-1:
            return # nothing to do
        
        self.drop_growbuffer()
        self.ydata = self.ydata[ibeg:iend].copy()
        self.tmin = self.tmin+ibeg*self.deltat
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        self.update_ids()
    
    def append(self, data):
        '''Append data to the end of the trace.'''
        
        assert self.ydata.dtype == data.dtype
        newlen = data.size + self.ydata.size
        if self._growbuffer is None or self._growbuffer.size < newlen:
            self._growbuffer = num.empty(newlen*2, dtype=self.ydata.dtype)
            self._growbuffer[:self.ydata.size] = self.ydata
        self._growbuffer[self.ydata.size:newlen] = data
        self.ydata = self._growbuffer[:newlen]
        self.tmax = self.tmin + (newlen-1)*self.deltat
        
    def chop(self, tmin, tmax, inplace=True, include_last=False, snap=(round,round), want_incomplete=True):
        if want_incomplete:
            if tmax <= self.tmin-self.deltat or self.tmax+self.deltat < tmin: 
                raise NoData()
        else:
            if tmin < self.tmin or self.tmax < tmax: 
                raise NoData()
        
        ibeg = max(0, t2ind(tmin-self.tmin,self.deltat, snap[0]))
        iplus = 0
        if include_last: iplus=1
        iend = min(len(self.ydata), t2ind(tmax-self.tmin,self.deltat, snap[1])+iplus)
        
        if ibeg >= iend: raise NoData()
        obj = self
        if not inplace:
            obj = self.copy(data=False)
       
        self.drop_growbuffer()
        obj.ydata = self.ydata[ibeg:iend].copy()
        obj.tmin = obj.tmin+ibeg*obj.deltat
        obj.tmax = obj.tmin+(len(obj.ydata)-1)*obj.deltat
        obj.update_ids()
        
        return obj
    
    def downsample(self, ndecimate, snap=False, initials=None, demean=True):
        newdeltat = self.deltat*ndecimate
        if snap:
            ilag = (math.ceil(self.tmin / newdeltat) * newdeltat - self.tmin)/self.deltat
            
        if snap and ilag > 0 and ilag < self.ydata.size:
            data = self.ydata.astype(num.float64)
            self.tmin += ilag*self.deltat
        else:
            data = self.ydata.astype(num.float64)
        
        if demean:
            data -= num.mean(data)
        
        result = util.decimate(data, ndecimate, ftype='fir', zi=initials)
        if initials is None:
            self.ydata, finals = result, None
        else:
            self.ydata, finals = result
            
        self.deltat = reuse(self.deltat*ndecimate)
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        self.update_ids()
        
        return finals
        
    def downsample_to(self, deltat, snap=False, allow_upsample_max=1, initials=None, demean=True):
        ratio = deltat/self.deltat
        rratio = round(ratio)
        if abs(rratio - ratio)/ratio > 0.0001:
            if allow_upsample_max <=1:
                raise util.UnavailableDecimation('ratio = %g' % ratio)
            else:
                deltat_inter = 1./util.lcm(1./self.deltat,1./deltat)
                upsratio = int(round(self.deltat/deltat_inter))
                if upsratio > allow_upsample_max:
                    raise util.UnavailableDecimation('ratio = %g' % ratio)
                
                if upsratio > 1:
                    self.drop_growbuffer()
                    ydata = self.ydata
                    self.ydata = num.zeros(ydata.size*upsratio-(upsratio-1), ydata.dtype)
                    self.ydata[::upsratio] = ydata
                    for i in range(1,upsratio):
                        self.ydata[i::upsratio] = float(i)/upsratio * ydata[:-1] + float(upsratio-i)/upsratio * ydata[1:]
                    self.deltat = self.deltat/upsratio
                
                    ratio = deltat/self.deltat
                    rratio = round(ratio)
            
        deci_seq = util.decitab(int(rratio))
        finals = []
        for i, ndecimate in enumerate(deci_seq):
             if ndecimate != 1:
                 xinitials = None
                 if initials is not None:
                     xinitials = initials[i]
                 finals.append(self.downsample(ndecimate, snap=snap, initials=xinitials, demean=demean))

        if initials is not None:
            return finals

    def nyquist_check(self, frequency, intro='Corner frequency', warn=True, raise_exception=False):
        if frequency >= 0.5/self.deltat:
            message = '%s (%g Hz) is equal to or higher than nyquist frequency (%g Hz). (Trace %s)' \
                    % (intro, frequency, 0.5/self.deltat, self.name())
            if warn:
                logger.warn(message)
            if raise_exception:
                raise AboveNyquist(message)
            
    def lowpass(self, order, corner):
        self.nyquist_check(corner, 'Corner frequency of lowpass')
        (b,a) = get_cached_filter_coefs(order, [corner*2.0*self.deltat], btype='low')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b,a, data)
        
    def highpass(self, order, corner):
        self.nyquist_check(corner, 'Corner frequency of highpass')
        (b,a) = get_cached_filter_coefs(order, [corner*2.0*self.deltat], btype='high')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b,a, data)
        
    def bandpass(self, order, corner_hp, corner_lp):
        self.nyquist_check(corner_hp, 'Lower corner frequency of bandpass')
        self.nyquist_check(corner_lp, 'Higher corner frequency of bandpass')
        (b,a) = get_cached_filter_coefs(order, [corner*2.0*self.deltat for corner in (corner_hp, corner_lp)], btype='band')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b,a, data)
        
    def get_cached_freqs(self, nf, deltaf):
        ck = (nf, deltaf)
        if ck not in Trace.cached_frequencies:
            Trace.cached_frequencies[ck] = num.arange(nf, dtype=num.float)*deltaf
        return Trace.cached_frequencies[ck]
        
    def bandpass_fft(self, corner_hp, corner_lp):
        n = len(self.ydata)
        n2 = nextpow2(n)
        data = num.zeros(n2, dtype=num.float64)
        data[:n] = self.ydata
        fdata = num.fft.rfft(data)
        freqs = self.get_cached_freqs(len(fdata), 1./(self.deltat*n2))
        fdata[0] = 0.0
        fdata *= num.logical_and(corner_hp < freqs, freqs < corner_lp)
        data = num.fft.irfft(fdata)
        self.drop_growbuffer()
        self.ydata = data[:n]
        
    def shift(self, tshift):
        self.tmin += tshift
        self.tmax += tshift
        self.update_ids()
        
    def snap(self):
        self.tmin = round(self.tmin/self.deltat)*self.deltat 
        self.tmax = self.tmin + (self.ydata.size-1)*self.deltat
        self.update_ids()

    def sta_lta_centered(self, tshort, tlong, quad=True):
    
        nshort = tshort/self.deltat
        nlong = tlong/self.deltat
    
        assert nshort < nlong
        if nlong > len(self.ydata):
            raise TraceTooShort('Samples in trace: %s, samples needed: %s' % (len(self.ydata), nlong))
         
        if quad:
            sqrdata = self.ydata**2
        else:
            sqrdata = self.ydata
    
        mavg_short = moving_avg(sqrdata,nshort)
        mavg_long = moving_avg(sqrdata,nlong)
    
        self.drop_growbuffer()
        self.ydata = num.maximum((mavg_short/mavg_long - 1.) * float(nshort)/float(nlong), 0.0)
        
    def peaks(self, threshold, tsearch):
        y = self.ydata
        above =  num.where(y > threshold, 1, 0)
        itrig_positions = num.nonzero((above[1:]-above[:-1])>0)[0]
        tpeaks = []
        apeaks = []
        for itrig_pos in itrig_positions:
            ibeg = max(0,itrig_pos - 0.5*tsearch/self.deltat)
            iend = min(len(self.ydata)-1, itrig_pos + 0.5*tsearch/self.deltat)
            ipeak = num.argmax(y[ibeg:iend])
            tpeak = self.tmin + (ipeak+ibeg)*self.deltat
            apeak = y[ibeg+ipeak]
            tpeaks.append(tpeak)
            apeaks.append(apeak)
            
        return tpeaks, apeaks
    
    def extend(self, tmin, tmax, fillmethod='zeros'):
        '''Extend trace to given span
        
        In:
            tmin, tmax -- new span
            fillmethod -- 'zeros' or 'repeat' 
        '''
        
        assert tmin <= self.tmin and tmax >= self.tmax
        
        nl = int(math.floor((self.tmin-tmin)/self.deltat))
        nh = int(math.floor((tmax-self.tmax)/self.deltat))
        self.tmin -= nl*self.deltat
        self.tmax += nh*self.deltat
        n = nl+self.ydata.size+nh

        data = num.zeros(n, dtype=self.ydata.dtype)
        data[nl:n-nh] = self.ydata
        if fillmethod == 'repeat' and self.ydata.size >= 1:
            data[:nl] = data[nl]
            data[n-nh:] = data[n-nh-1]
            
        self.drop_growbuffer()
        self.ydata = data
        
        self.update_ids()
     
    def transfer(self, tfade, freqlimits, transfer_function=None, cut_off_fading=True):
        '''Return new trace with transfer function applied.
        
        tfade -- rise/fall time in seconds of taper applied in timedomain at both ends of trace.
        freqlimits -- 4-tuple with corner frequencies in Hz.
        transfer_function -- FrequencyResponse object; must provide a method 'evaluate(freqs)', which returns the
                             transfer function coefficients at the frequencies 'freqs'.
        cut_off_fading -- cut off rise/fall interval in output trace.
        '''
    
        if transfer_function is None:
            transfer_function = FrequencyResponse()
    
        if self.tmax - self.tmin <= tfade*2.:
            raise TraceTooShort('Trace %s.%s.%s.%s too short for fading length setting. trace length = %g, fading length = %g' % (self.nslc_id + (self.tmax-self.tmin, tfade)))

        ndata = self.ydata.size
        ntrans = nextpow2(ndata*1.2)
        coefs = self._get_tapered_coefs(ntrans, freqlimits, transfer_function)
        
        data = self.ydata
        data_pad = num.zeros(ntrans, dtype=num.float)
        data_pad[:ndata]  = data - data.mean()
        data_pad[:ndata] *= costaper(0.,tfade, self.deltat*(ndata-1)-tfade, self.deltat*ndata, ndata, self.deltat)
        fdata = num.fft.rfft(data_pad)
        fdata *= coefs
        ddata = num.fft.irfft(fdata)
        output = self.copy()
        output.ydata = ddata[:ndata]
        if cut_off_fading:
            try:
                output.chop(output.tmin+tfade, output.tmax-tfade, inplace=True)
            except NoData:
                raise TraceTooShort('Trace %s.%s.%s.%s too short for fading length setting. trace length = %g, fading length = %g' % (self.nslc_id + (self.tmax-self.tmin, tfade)))
        else:
            output.ydata = output.ydata.copy()
        return output
        
    def spectrum(self, pad_to_pow2=False, tfade=None):
        ndata = self.ydata.size
        
        if pad_to_pow2:
            ntrans = nextpow2(ndata)
        else:
            ntrans = ndata

        if tfade is None:
            ydata = self.ydata
        else:
            ydata = self.ydata * costaper(0., tfade, self.deltat*(ndata-1)-tfade, self.deltat*ndata, ndata, self.deltat)
            
        fydata = num.fft.rfft(ydata, ntrans)
        df = 1./(ntrans*self.deltat)
        fxdata = num.arange(len(fydata))*df
        return fxdata, fydata
        
    def _get_tapered_coefs(self, ntrans, freqlimits, transfer_function):
    
        deltaf = 1./(self.deltat*ntrans)
        nfreqs = ntrans/2 + 1
        transfer = num.ones(nfreqs, dtype=num.complex)
        hi = snapper(nfreqs, deltaf)
        a,b,c,d = freqlimits
        freqs = num.arange(hi(d)-hi(a), dtype=num.float)*deltaf + hi(a)*deltaf
        transfer[hi(a):hi(d)] = transfer_function.evaluate(freqs)
        
        tapered_transfer = costaper(a,b,c,d, nfreqs, deltaf)*transfer
        tapered_transfer[0] = 0.0 # don't introduce static offsets
        return tapered_transfer
        
    def fill_template(self, template, **additional):
        params = dict(zip( ('network', 'station', 'location', 'channel'), self.nslc_id))
        params['tmin'] = util.time_to_str(self.tmin, format='%Y-%m-%d_%H-%M-%S')
        params['tmax'] = util.time_to_str(self.tmax, format='%Y-%m-%d_%H-%M-%S')
        params['tmin_ms'] = util.time_to_str(self.tmin, format='%Y-%m-%d_%H-%M-%S.3FRAC')
        params['tmax_ms'] = util.time_to_str(self.tmax, format='%Y-%m-%d_%H-%M-%S.3FRAC')
        params['tmin_us'] = util.time_to_str(self.tmin, format='%Y-%m-%d_%H-%M-%S.6FRAC')
        params['tmax_us'] = util.time_to_str(self.tmax, format='%Y-%m-%d_%H-%M-%S.6FRAC')
        params.update(additional)
        return template % params

cached_coefficients = {}
def get_cached_filter_coefs(order, corners, btype):
    ck = (order, tuple(corners), btype)
    if ck not in cached_coefficients:
        if len(corners) == 0:
            cached_coefficients[ck] = signal.butter(order, corners[0], btype=btype)
        else:
            cached_coefficients[ck] = signal.butter(order, corners, btype=btype)

    return cached_coefficients[ck]
    
    
