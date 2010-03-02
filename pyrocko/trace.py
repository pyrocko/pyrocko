
import util, time, math, copy, logging
import numpy as num
from util import reuse
from scipy import signal

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
    
def degapper(in_traces, maxgap=5, fillmethod='interpolate'):
    
    '''Try to connect traces and remove gaps.
    
    This method will combine adjacent traces, which match in their network, 
    station, location and channel attributes. Overlapping parts will be removed.
    
    Arguments:
    
       in_traces:   input traces, must be sorted by their full_id attribute.
       maxgap:      maximum number of samples to interpolate.
       fillmethod:  what to put into the gaps: 'interpolate' or 'zeros'.
       
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
            if abs(dist - idist) > 0.05:
                logger.warn('cannot degap traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
            else:
                idist = int(round(dist))
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
                        a.ydata = num.concatenate((a.ydata[:idist-1], b.ydata))
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

class CannotRotate(Exception):
    pass

def rotate(traces, azimuth, in_channels, out_channels):
    phi = azimuth/180.*math.pi
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    rotated = []
    for a in traces:
        for b in traces:
            if ( (a.channel, b.channel) == in_channels and
                 a.nslc_id[:3] == b.nslc_id[:3] and
                 abs(a.deltat-b.deltat) < a.deltat*0.001 ):
                
                tmin = max(a.tmin, b.tmin)
                tmax = min(a.tmax, b.tmax)
                
                if tmin < tmax:
                    ac = a.chop(tmin, tmax, inplace=False)
                    bc = b.chop(tmin, tmax, inplace=False)
                    if (ac.tmin - bc.tmin) > ac.deltat*0.01:
                        logger.warn('cannot rotate traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
                        continue
                    
                    ac.set_ydata(ac.get_ydata()*cphi+bc.get_ydata()*sphi)
                    bc.set_ydata(-ac.get_ydata()*sphi+bc.get_ydata()*cphi)

                    ac.set_codes(channel=out_channels[0])
                    bc.set_codes(channel=out_channels[1])
                    rotated.append(ac)
                    rotated.append(bc)
                    
    return rotated
            
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

def t2ind(t,tdelta):
    return int(round(t/tdelta))

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
        
        
class IntegrationResponse(FrequencyResponse):
    def __init__(self, gain=1.0):
        self._gain = gain
        
    def evaluate(self, freqs):
        return (self._gain/(1.0j * 2. * num.pi)) / freqs

class DifferentiationResponse(FrequencyResponse):
    def __init__(self, gain=1.0):
        self._gain = gain
        
    def evaluate(self, freqs):
        return self._gain * 1.0j * 2. * num.pi * freqs

class NoData(Exception):
    pass

class Trace(object):
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
        
    def __eq__(self, other):
        return (self.network == other.network and
                self.station == other.station and
                self.location == other.location and
                self.channel == other.channel and
                self.deltat == other.deltat and
                abs(self.tmin-other.tmin) < self.deltat*0.001 and
                abs(self.tmax-other.tmax) < self.deltat*0.001 and
                num.all(self.ydata == other.ydata))
                
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
        self.ydata = new_ydata
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        
    def drop_data(self):
        self.ydata = None
    
    def copy(self, data=True):
        tracecopy = copy.copy(self)
        if data:
            tracecopy.ydata = self.ydata.copy()
        tracecopy.meta = copy.deepcopy(self.meta)
        return tracecopy
    
    def append(self, data):
        assert self.ydata.dtype == data.dtype
        newlen = data.size + self.ydata.size
        if not hasattr(self, 'growbuffer') or self.growbuffer.size < newlen:
            self.growbuffer = num.empty(newlen*2, dtype=self.ydata.dtype)
            self.growbuffer[:self.ydata.size] = self.ydata
        self.growbuffer[self.ydata.size:newlen] = data
        self.ydata = self.growbuffer[:newlen]
        self.tmax = self.tmin + (newlen-1)*self.deltat
        
    def chop(self, tmin, tmax, inplace=True, include_last=False):
        if (tmax <= self.tmin or self.tmax < tmin): raise NoData()
        ibeg = max(0, t2ind(tmin-self.tmin,self.deltat))
        iplus = 0
        if include_last: iplus=1
        iend = min(len(self.ydata), t2ind(tmax-self.tmin,self.deltat)+iplus)
        
        obj = self
        if not inplace:
            obj = self.copy(data=False)
        
        obj.ydata = self.ydata[ibeg:iend].copy()
        obj.tmin = obj.tmin+ibeg*obj.deltat
        obj.tmax = obj.tmin+(len(obj.ydata)-1)*obj.deltat
        obj.update_ids()
        return obj
    
    def downsample(self, ndecimate, snap=False):
        newdeltat = self.deltat*ndecimate
        if snap:
            ilag = (math.ceil(self.tmin / newdeltat) * newdeltat - self.tmin)/self.deltat
            
        if snap and ilag > 0 and ilag < self.ydata.size:
            data = self.ydata.astype(num.float64)
            self.tmin += ilag*self.deltat
        else:
            data = self.ydata.astype(num.float64)
            
        data -= num.mean(data)
        self.ydata = util.decimate(data, ndecimate, ftype='fir')
        self.deltat = reuse(self.deltat*ndecimate)
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        self.update_ids()
        
    def downsample_to(self, deltat, snap=False):
        ratio = deltat/self.deltat
        rratio = round(ratio)
        if abs(rratio - ratio) > 0.0001: raise util.UnavailableDecimation('ratio = %g' % ratio)
        deci_seq = util.decitab(int(rratio))
        for ndecimate in deci_seq:
             if ndecimate != 1:
                self.downsample(ndecimate, snap=snap)
            
    def lowpass(self, order, corner):
        (b,a) = signal.butter(order, corner*2.0*self.deltat, btype='low')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = signal.lfilter(b,a, data)
        
    def highpass(self, order, corner):
        (b,a) = signal.butter(order, corner*2.0*self.deltat, btype='high')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = signal.lfilter(b,a, data)
        
    def bandpass(self, order, corner_hp, corner_lp):
        (b,a) = signal.butter(order, [corner*2.0*self.deltat for corner in (corner_hp, corner_lp)], btype='band')
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = signal.lfilter(b,a, data)
        
    def bandpass_fft(self, corner_hp, corner_lp):
        data = self.ydata.astype(num.float64)
        n = len(data)
        fdata = num.fft.rfft(data)
        nf = len(fdata)
        df = 1./(n*self.deltat)
        freqs = num.arange(nf)*df
        fdata *= num.logical_and(corner_hp < freqs, freqs < corner_lp)
        data = num.fft.irfft(fdata,n)
        assert len(data) == n
        self.ydata = data
        
    def shift(self, tshift):
        self.tmin += tshift
        self.tmax += tshift
        self.update_ids()
        
    def sta_lta_centered(self, tshort, tlong, quad=True):
    
        nshort = tshort/self.deltat
        nlong = tlong/self.deltat
    
        if quad:
            sqrdata = self.ydata**2
        else:
            sqrdata = self.ydata
    
        mavg_short = moving_avg(sqrdata,nshort)
        mavg_long = moving_avg(sqrdata,nlong)
    
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
            raise TraceTooShort('trace too short for fading length setting. trace length = %g, fading length = %g' % (self.tmax-self.tmin, tfade))

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
            output = output.chop(output.tmin+tfade, output.tmax-tfade)
        else:
            output.ydata = output.ydata.copy()
        return output
        
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
        
    def fill_template(self, template):
        params = dict(zip( ('network', 'station', 'location', 'channel'), self.nslc_id))
        params['tmin'] = util.gmctime_fn(self.tmin)
        params['tmax'] = util.gmctime_fn(self.tmax)
        return template % params
        
    def __str__(self):
        s = 'Trace (%s, %s, %s, %s)\n' % self.nslc_id
        s += '  timerange: %s - %s\n' % (util.gmctime(self.tmin), util.gmctime(self.tmax))
        s += '  delta t: %g\n' % self.deltat
        return s
