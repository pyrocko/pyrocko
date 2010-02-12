
import util, time
import numpy as num
from util import reuse


class NoData(Exception):
    pass

class Trace(object):
    def __init__(self, network='', station='STA', location='', channel='', 
                 tmin=0., tmax=None, deltat=1., ydata=None, mtime=None, meta=None):
    
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
                
    def is_relevant(self, tmin, tmax, selector=None):
        return  not (tmax <= self.tmin or self.tmax < tmin) and (selector is None or selector(self))

    def update_ids(self):
        self.full_id = (self.network,self.station,self.location,self.channel,self.tmin)
        self.nslc_id = reuse((self.network,self.station,self.location,self.channel))

    def set_mtime(self, mtime):
        self.mtime = mtime

    def get_xdata(self):
        return self.tmin + num.arange(len(self.ydata), dtype=num.float64) * self.deltat

    def get_ydata(self):
        if self.ydata is None: raise NoData()
        return self.ydata
        
    def drop_data(self):
        self.ydata = None
    
    def copy(self, data=True):
        tracecopy = copy.copy(self)
        if copydata:
            tracecopy.ydata = self.ydata.copy()
        tracecopy.meta = copy.deepcopy(self.meta)
        return tracecopy
        
    def chop(self, tmin, tmax, inplace=True):
        if (tmax <= self.tmin or self.tmax < tmin): raise NoData()
        ibeg = max(0, t2ind(tmin-self.tmin,self.deltat))
        iend = min(len(self.ydata), t2ind(tmax-self.tmin,self.deltat))
        
        obj = self
        if not inplace:
            obj = self.copy(data=False)
        
        obj.ydata = self.ydata[ibeg:iend].copy()
        obj.tmin = obj.tmin+ibeg*obj.deltat
        obj.tmax = obj.tmin+(len(obj.ydata)-1)*obj.deltat
        
        return obj
    
    def downsample(self, ndecimate):
        data = self.ydata.astype(num.float64)
        data -= num.mean(data)
        self.ydata = decimate(data, ndecimate, ftype='fir')
        self.deltat = reuse(self.deltat*ndecimate)
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        
    def downsample_to(self, deltat):
        ratio = deltat/self.deltat
        rratio = round(ratio)
        if abs(rratio - ratio) > 0.0001: raise UnavailableDecimation('ratio = %g' % ratio)
        deci_seq = decitab(int(rratio))
        for ndecimate in deci_seq:
             if ndecimate != 1:
                self.downsample(ndecimate)
            
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
