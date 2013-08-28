'''This module provides basic signal processing for seismic traces.


'''

import util, evalresp
import time, math, copy, logging, sys
import numpy as num
from util import reuse, hpfloat
from scipy import signal
from pyrocko import model, orthodrome

logger = logging.getLogger('pyrocko.trace')


class Trace(object):
    
    '''Create new trace object.

    A ``Trace`` object represents a single continuous strip of evenly sampled
    time series data.  It is built from a 1D NumPy array containing the data
    samples and some attributes describing its beginning  and ending time, its
    sampling rate and four string identifiers (its network, station, location
    and channel code).

    :param network:  network code
    :param station:  station code
    :param location:  location code
    :param channel:  channel code
    :param tmin:  system time of first sample in [s]
    :param tmax:  system time of last sample in [s] (if set to ``None`` it is computed from length of *ydata*)
    :param deltat:  sampling interval in [s]
    :param ydata:  1D numpy array with data samples (can be ``None`` when *tmax* is not ``None``)
    :param mtime:  optional modification time 
    :param meta:  additional meta information (not used, but maintained by the library)

    The length of the network, station, location and channel codes is not resricted by this software,
    but data formats like SAC, Mini-SEED or GSE have different limits on the lengths of these codes. The codes set here
    are silently truncated when the trace is stored
    '''

    cached_frequencies = {}
        
    def __init__(self, network='', station='STA', location='', channel='', 
                 tmin=0., tmax=None, deltat=1., ydata=None, mtime=None, meta=None):
    
        self._growbuffer = None
        
        if deltat < 0.001:
            tmin = hpfloat(tmin)
            if tmax is not None:
                tmax = hpfloat(tmax)
    
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
        self._update_ids()
        
    def __str__(self):
        fmt = min(9, max(0, -int(math.floor(math.log10(self.deltat)))))
        s = 'Trace (%s, %s, %s, %s)\n' % self.nslc_id
        s += '  timerange: %s - %s\n' % (util.time_to_str(self.tmin, format=fmt), util.time_to_str(self.tmax, format=fmt))
        s += '  delta t: %g\n' % self.deltat
        if self.meta:
            for k in sorted(self.meta.keys()):
                s += '  %s: %s\n' % (k,self.meta[k])
        return s
       
    def __getstate__(self):
        return (self.network, self.station, self.location, self.channel, self.tmin, self.tmax, self.deltat, self.mtime)

    def __setstate__(self, state):
        self.network, self.station, self.location, self.channel, self.tmin, self.tmax, self.deltat, self.mtime = state
        self.ydata = None
        self.meta = None
        self._growbuffer = None
        self._update_ids()

    def name(self):
        '''Get a short string description.'''

        s = '%s.%s.%s.%s, %s, %s' % (self.nslc_id + (util.time_to_str(self.tmin), util.time_to_str(self.tmax)))
        return s
        
    def __eq__(self, other):

        return (self.network == other.network and
                self.station == other.station and
                self.location == other.location and
                self.channel == other.channel and
                abs(self.deltat - other.deltat) < (self.deltat + other.deltat)*1e-6 and
                abs(self.tmin-other.tmin) < self.deltat*0.01 and
                abs(self.tmax-other.tmax) < self.deltat*0.01 and
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
        '''Value of trace between supporting points through linear interpolation.
        
        :param t: time instant
        :param clip: whether to clip indices to trace ends
        '''

        t0, y0 = self(t, clip=clip, snap=math.floor)
        t1, y1 = self(t, clip=clip, snap=math.ceil)
        if t0 == t1:
            return y0
        else:
            return y0+(t-t0)/(t1-t0)*(y1-y0)
        
    def index_clip(self, i):
        '''Clip index to valid range.'''

        return min(max(0,i), self.ydata.size)

        
    def add(self, other, interpolate=True):
        '''Add values of other trace (self += other).
        
        Add values of *other* trace to the values of *self*, where it
        intersects with *other*.  This method does not change the extent of
        *self*. If *interpolate* is ``True`` (the default), the values of
        *other* to be added are interpolated at sampling instants of *self*.
        Linear interpolation is performed. In this case the sampling rate of
        *other* must be equal to or lower than that of *self*.  If
        *interpolate* is ``False``, the sampling rates of the two traces must
        match.
        '''
        
        if interpolate:
            assert self.deltat <= other.deltat or same_sampling_rate(self,other)
            other_xdata = other.get_xdata()
            xdata = self.get_xdata()
            xmin, xmax = other_xdata[0], other_xdata[-1]
            self.ydata += num.interp(xdata, other_xdata, other.ydata, left=0., right=0.)
        else:
            assert self.deltat == other.deltat
            ioff = int(round((other.tmin-self.tmin)/self.deltat))
            ibeg = max(0, ioff)
            iend = min(self.data_len(), ioff+other.data_len())
            self.ydata[ibeg:iend] += other.ydata[ibeg-ioff:iend-ioff]

    def mult(self, other, interpolate=True):
        '''Muliply with values of other trace (self \*= other).
        
        Multiply values of *other* trace to the values of *self*, where it
        intersects with *other*.  This method does not change the extent of
        *self*. If *interpolate* is ``True`` (the default), the values of
        *other* to be multiplied are interpolated at sampling instants of *self*.
        Linear interpolation is performed. In this case the sampling rate of
        *other* must be equal to or lower than that of *self*.  If
        *interpolate* is ``False``, the sampling rates of the two traces must
        match.
        '''

        if interpolate:
            assert self.deltat <= other.deltat or same_sampling_rate(self,other)
            other_xdata = other.get_xdata()
            xdata = self.get_xdata()
            xmin, xmax = other_xdata[0], other_xdata[-1]
            self.ydata *= num.interp(xdata, other_xdata, other.ydata, left=0., right=0.)
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
            
            self.ydata[ibeg1:iend1] *= other.ydata[ibeg2:iend2]
    
    def max(self):
        '''Get time and value of data maximum.'''

        i = num.argmax(self.ydata)
        return self.tmin + i*self.deltat, self.ydata[i]

    def min(self):
        '''Get time and value of data minimum.'''

        i = num.argmin(self.ydata)
        return self.tmin + i*self.deltat, self.ydata[i]

    def absmax(self):
        '''Get time and value of maximum of the absolute of data.'''
        tmi, mi = self.min()
        tma, ma = self.max()
        if abs(mi) > abs(ma):
            return tmi, abs(mi)
        else:
            return tma, abs(ma)

    def set_codes(self, network=None, station=None, location=None, channel=None):
        '''Set network, station, location, and channel codes.'''
        if network is not None:
            self.network = network
        if station is not None:
            self.station = station
        if location is not None:
            self.location = location
        if channel is not None:
            self.channel = channel
        
        self._update_ids()

    def set_network(self, network):
        self.network = network
        self._update_ids()

    def set_station(self, station):
        self.station = station
        self._update_ids()

    def set_location(self, location):
        self.location = location
        self._update_ids()

    def set_channel(self, channel):
        self.channel = channel
        self._update_ids()

    def overlaps(self, tmin, tmax):
        '''Check if trace has overlap with a given time span.'''
        return not (tmax < self.tmin or self.tmax < tmin)
           
    def is_relevant(self, tmin, tmax, selector=None):
        '''Check if trace has overlap with a given time span and matches a condition callback. (internal use)'''
        return  not (tmax <= self.tmin or self.tmax < tmin) and (selector is None or selector(self))

    def _update_ids(self):
        '''Update dependent ids.'''
        self.full_id = (self.network,self.station,self.location,self.channel,self.tmin)
        self.nslc_id = reuse((self.network,self.station,self.location,self.channel))

    def set_mtime(self, mtime):
        '''Set modification time of the trace.'''
        self.mtime = mtime

    def get_xdata(self):
        '''Create array for time axis.'''
        if self.ydata is None: raise NoData()
        return self.tmin + num.arange(len(self.ydata), dtype=num.float64) * self.deltat

    def get_ydata(self):
        '''Get data array.'''
        if self.ydata is None: raise NoData()
        return self.ydata
        
    def set_ydata(self, new_ydata):
        '''Replace data array.'''
        self.drop_growbuffer()
        self.ydata = new_ydata
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat

    def data_len(self):
        if self.ydata is not None:
            return self.ydata.size
        else:
            return int(round((self.tmax-self.tmin)/self.deltat)) + 1

    def drop_data(self):
        '''Forget data, make dataless trace.'''
        self.drop_growbuffer()
        self.ydata = None
   
    def drop_growbuffer(self):
        '''Detach the traces grow buffer.'''
        self._growbuffer = None

    def copy(self, data=True):
        '''Make a deep copy of the trace.'''
        tracecopy = copy.copy(self)
        self.drop_growbuffer()
        if data:
            tracecopy.ydata = self.ydata.copy()
        tracecopy.meta = copy.deepcopy(self.meta)
        return tracecopy
    
    def crop_zeros(self):
        '''Remove any zeros at beginning and end.'''
        
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
        self._update_ids()
    
    def append(self, data):
        '''Append data to the end of the trace.
        
        To make this method efficient when successively very few or even single samples are appended,
        a larger grow buffer is allocated upon first invocation. The traces data is then changed to
        be a view into the currently filled portion of the grow buffer array.'''
        
        assert self.ydata.dtype == data.dtype
        newlen = data.size + self.ydata.size
        if self._growbuffer is None or self._growbuffer.size < newlen:
            self._growbuffer = num.empty(newlen*2, dtype=self.ydata.dtype)
            self._growbuffer[:self.ydata.size] = self.ydata
        self._growbuffer[self.ydata.size:newlen] = data
        self.ydata = self._growbuffer[:newlen]
        self.tmax = self.tmin + (newlen-1)*self.deltat
        
    def chop(self, tmin, tmax, inplace=True, include_last=False, snap=(round,round), want_incomplete=True):
        '''Cut the trace to given time span.

        If the *inplace* argument is True (the default) the trace is cut in
        place, otherwise a new trace with the cut part is returned.  By
        default, the indices where to start and end the trace data array are
        determined by rounding of *tmin* and *tmax* to sampling instances using
        Python's :py:func:`round` function. This behaviour can be changed with
        the *snap* argument, which takes a tuple of two functions (one for the
        lower and one for the upper end) to be used instead of
        :py:func:`round`.  The last sample is by default not included unless
        *include_last* is set to True.  If the given time span exceeds the
        available time span of the trace, the available part is returned,
        unless *want_incomplete* is set to False - in that case, a
        :py:exc:`NoData` exception is raised. This exception is always
        raised, when the requested time span does dot overlap with the trace's
        time span.
        '''
        
        if want_incomplete:
            if tmax <= self.tmin-self.deltat or self.tmax+self.deltat < tmin: 
                raise NoData()
        else:
            if tmin < self.tmin or self.tmax < tmax: 
                raise NoData()
        
        ibeg = max(0, t2ind(tmin-self.tmin,self.deltat, snap[0]))
        iplus = 0
        if include_last: iplus=1

        iend = min(self.data_len(), t2ind(tmax-self.tmin,self.deltat, snap[1])+iplus)
        
        if ibeg >= iend: raise NoData()
        obj = self
        if not inplace:
            obj = self.copy(data=False)
       
        self.drop_growbuffer()
        if self.ydata is not None:
            obj.ydata = self.ydata[ibeg:iend].copy()
        else:
            obj.ydata = None
        
        obj.tmin = obj.tmin+ibeg*obj.deltat
        obj.tmax = obj.tmin+((iend-ibeg)-1)*obj.deltat
            
        obj._update_ids()
    
        
        return obj
    
    def downsample(self, ndecimate, snap=False, initials=None, demean=True):
        '''Downsample trace by a given integer factor.
        
        :param ndecimate: decimation factor, avoid values larger than 8
        :param snap: whether to put the new sampling instances closest to multiples of the sampling rate.
        :param initials: ``None``, ``True``, or initial conditions for the anti-aliasing filter, obtained from a
            previous run. In the latter two cases the final state of the filter is returned instead of ``None``.
        :param demean: whether to demean the signal before filtering.
        '''

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
        self._update_ids()
        
        return finals
        
    def downsample_to(self, deltat, snap=False, allow_upsample_max=1, initials=None, demean=True):
        '''Downsample to given sampling rate.

        Tries to downsample the trace to a target sampling interval of
        *deltat*. This runs the :py:meth:`Trace.downsample` one or several times. If
        allow_upsample_max is set to a value larger than 1, intermediate
        upsampling steps are allowed, in order to increase the number of
        possible downsampling ratios.
        '''

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

    def resample(self, deltat):

        ndata = self.ydata.size
        ntrans = nextpow2(ndata)
        fntrans2 = ntrans * self.deltat/deltat
        ntrans2 = int(round(fntrans2))
        deltat2 = self.deltat * float(ntrans)/float(ntrans2)
        ndata2 = int(round(ndata*self.deltat/deltat2))
        if abs(fntrans2 - ntrans2) > 1e-7:
            logger.warn('resample: requested deltat %g could not be matched exactly: %g' % (deltat, deltat2))
        
        data = self.ydata
        data_pad = num.zeros(ntrans, dtype=num.float)
        data_pad[:ndata]  = data
        fdata = num.fft.rfft(data_pad)
        fdata2 = num.zeros((ntrans2+1)/2, dtype=fdata.dtype)
        n = min(fdata.size,fdata2.size)
        fdata2[:n] = fdata[:n]
        data2 = num.fft.irfft(fdata2)
        data2 = data2[:ndata2]
        data2 *= float(ntrans2) / float(ntrans)
        self.deltat = deltat2
        self.set_ydata(data2)

    def resample_simple(self, deltat):
        tyear = 3600*24*365.

        if deltat == self.deltat:
            return

        if abs(self.deltat - deltat) * tyear/deltat < deltat:
            logger.warn('resample_simple: less than one sample would have to be inserted/deleted per year. Doing nothing.')
            return

        ninterval = int(round(deltat / (self.deltat - deltat)))
        if abs(ninterval) < 20:
            logger.error('resample_simple: sample insertion/deletion interval less than 20. results would be erroneous.')
            raise ResamplingFailed()

        delete = False
        if ninterval < 0:
            ninterval = - ninterval
            delete = True
        
        tyearbegin = util.year_start(self.tmin)
        
        nmin = int(round((self.tmin - tyearbegin)/deltat))

        ibegin = (((nmin-1)/ninterval)+1) * ninterval - nmin
        nindices = (len(self.ydata) - ibegin - 1) / ninterval + 1
        if nindices > 0:
            indices = ibegin + num.arange(nindices) * ninterval
            data_split = num.split(self.ydata, indices)
            data = []
            for l, h in zip(data_split[:-1], data_split[1:]):
                if delete:
                    l = l[:-1]
                    
                data.append(l)
                if not delete:
                    if l.size == 0:
                        v = h[0]
                    else:
                        v = 0.5*(l[-1] + h[0])
                    data.append(num.array([v], dtype=l.dtype))

            data.append(data_split[-1])

            ydata_new = num.concatenate( data )

            self.tmin = tyearbegin + nmin * deltat
            self.deltat = deltat
            self.set_ydata(ydata_new)

    def nyquist_check(self, frequency, intro='Corner frequency', warn=True, raise_exception=False):
        '''Check if a given frequency is above the Nyquist frequency of the trace.

        :param intro: string used to introduce the warning/error message
        :param warn: whether to emit a warning
        :param raise_exception: whether to raise an :py:exc:`AboveNyquist` exception.
        '''

        if frequency >= 0.5/self.deltat:
            message = '%s (%g Hz) is equal to or higher than nyquist frequency (%g Hz). (Trace %s)' \
                    % (intro, frequency, 0.5/self.deltat, self.name())
            if warn:
                logger.warn(message)
            if raise_exception:
                raise AboveNyquist(message)
            
    def lowpass(self, order, corner, nyquist_warn=True, nyquist_exception=False, demean=True):
        '''Apply Butterworth lowpass to the trace.
        
        :param order: order of the filter
        :param corner: corner frequency of the filter

        Mean is removed before filtering.
        '''
        self.nyquist_check(corner, 'Corner frequency of lowpass', nyquist_warn, nyquist_exception)
        (b,a) = _get_cached_filter_coefs(order, [corner*2.0*self.deltat], btype='low')
        if len(a) != order+1 or len(b) != order+1:
            logger.warn('Erroneous filter coefficients returned by scipy.signal.butter(). You may need to downsample the signal before filtering.')

        data = self.ydata.astype(num.float64)
        if demean:
            data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b,a, data)
        
    def highpass(self, order, corner, nyquist_warn=True, nyquist_exception=False, demean=True):
        '''Apply butterworth highpass to the trace.

        :param order: order of the filter
        :param corner: corner frequency of the filter
        
        Mean is removed before filtering.
        '''

        self.nyquist_check(corner, 'Corner frequency of highpass', nyquist_warn, nyquist_exception)
        (b,a) = _get_cached_filter_coefs(order, [corner*2.0*self.deltat], btype='high')
        data = self.ydata.astype(num.float64)
        if len(a) != order+1 or len(b) != order+1:
            logger.warn('Erroneous filter coefficients returned by scipy.signal.butter(). You may need to downsample the signal before filtering.')
        if demean:
            data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b,a, data)
        
    def bandpass(self, order, corner_hp, corner_lp, demean=True):
        '''Apply butterworth bandpass to the trace.
        
        :param order: order of the filter
        :param corner_hp: lower corner frequency of the filter
        :param corner_lp: upper corner frequency of the filter

        Mean is removed before filtering.
        '''

        self.nyquist_check(corner_hp, 'Lower corner frequency of bandpass')
        self.nyquist_check(corner_lp, 'Higher corner frequency of bandpass')
        (b,a) = _get_cached_filter_coefs(order, [corner*2.0*self.deltat for corner in (corner_hp, corner_lp)], btype='band')
        data = self.ydata.astype(num.float64)
        if demean:
            data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b,a, data)
    
    def abshilbert(self):
        self.drop_growbuffer()
        self.ydata = num.abs(hilbert(self.ydata))
    
    def envelope(self):
        self.drop_growbuffer()
        self.ydata = num.sqrt(self.ydata**2 + hilbert(self.ydata)**2)

    def taper(self, taperer):
        taperer(self.ydata, self.tmin, self.deltat)
    
    def whiten(self, order=6):
        '''Whiten signal in time domain using autoregression and recursive filter.
        
        :param order: order of the autoregression process
        '''
        
        b,a = self.whitening_coefficients(order)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b,a, self.ydata)

    def whitening_coefficients(self, order=6):
        ar = yulewalker(self.ydata, order)
        b, a = [1.] + ar.tolist(), [1.]
        return b, a

    def ampspec_whiten(self, width, td_taper='auto', fd_taper='auto', pad_to_pow2=True, demean=True):
        '''Whiten signal via frequency domain using moving average on amplitude spectra.
        
        :param width: width of smoothing kernel [Hz]
        :param td_taper: time domain taper, object of type :py:class:`Taper` or ``None`` or ``'auto'``.
        :param fd_taper: frequency domain taper, object of type :py:class:`Taper` or ``None`` or ``'auto'``.
        :param pad_to_pow2: whether to pad the signal with zeros up to a length of 2^n
        :param demean: whether to demean the signal before tapering
        
        The signal is first demeaned and then tapered using *td_taper*. Then,
        the spectrum is calculated and inversely weighted with a smoothed
        version of its amplitude spectrum. A moving average is used for the
        smoothing. The smoothed spectrum is then tapered using *fd_taper*.
        Finally, the smoothed and tapered spectrum is back-transformed into the
        time domain.

        If *td_taper* is set to ``'auto'``, ``CosFader(1.0/width)`` is used. If
        *fd_taper* is set to ``'auto'``, ``CosFader(width)`` is used.

        '''
        ndata = self.data_len()
        
        if pad_to_pow2:
            ntrans = nextpow2(ndata)
        else:
            ntrans = ndata

        df = 1./(ntrans*self.deltat)
        nw = int(round(width/df))
        if ndata/2+1 <= nw:
            raise TraceTooShort('Samples in trace: %s, samples needed: %s' % (ndata, nw))

        if td_taper == 'auto':
            td_taper = CosFader(1./width)

        if fd_taper == 'auto':
            fd_taper = CosFader(width)
        
        if td_taper:
            self.taper(td_taper)

        ydata = self.get_ydata().astype(num.float)
        if demean:
            ydata -= ydata.mean()
            
        spec = num.fft.rfft(ydata, ntrans)
        
        amp = num.abs(spec)
        nspec = amp.size
        csamp = num.cumsum(amp)
        amp_smoothed = num.empty(nspec, dtype=csamp.dtype)
        n1, n2 = nw/2, nw/2 + nspec - nw
        amp_smoothed[n1:n2] = (csamp[nw:] - csamp[:-nw]) / nw
        amp_smoothed[:n1] = amp_smoothed[n1]
        amp_smoothed[n2:] = amp_smoothed[n2-1]
       
        denom = amp_smoothed * amp
        numer = amp
        eps = num.mean(denom) * 1e-9
        if eps == 0.0:
            eps = 1e-9
        
        numer += eps
        denom += eps
        spec *= numer/denom
        
        if fd_taper:
            fd_taper(spec, 0., df)

        ydata =  num.fft.irfft(spec)
        self.set_ydata(ydata[:ndata])

    def _get_cached_freqs(self, nf, deltaf):
        ck = (nf, deltaf)
        if ck not in Trace.cached_frequencies:
            Trace.cached_frequencies[ck] = num.arange(nf, dtype=num.float)*deltaf
        return Trace.cached_frequencies[ck]
        
    def bandpass_fft(self, corner_hp, corner_lp):
        '''Apply boxcar bandbpass to trace (in spectral domain).'''

        n = len(self.ydata)
        n2 = nextpow2(n)
        data = num.zeros(n2, dtype=num.float64)
        data[:n] = self.ydata
        fdata = num.fft.rfft(data)
        freqs = self._get_cached_freqs(len(fdata), 1./(self.deltat*n2))
        fdata[0] = 0.0
        fdata *= num.logical_and(corner_hp < freqs, freqs < corner_lp)
        data = num.fft.irfft(fdata)
        self.drop_growbuffer()
        self.ydata = data[:n]
        
    def shift(self, tshift):
        '''Time shift the trace.'''

        self.tmin += tshift
        self.tmax += tshift
        self._update_ids()
        
    def snap(self):
        '''Shift trace samples to nearest even multiples of the sampling rate.'''

        self.tmin = round(self.tmin/self.deltat)*self.deltat 
        self.tmax = self.tmin + (self.ydata.size-1)*self.deltat
        self._update_ids()

    def sta_lta_centered(self, tshort, tlong, quad=True, scalingmethod=1):
        '''Run special STA/LTA filter where the short time window is centered on the long time window.

        :param tshort: length of short time window in [s]
        :param tlong: length of long time window in [s]
        :param quad: whether to square the data prior to applying the STA/LTA filter
        :param scalingmethod: integer key to select how output values are scaled / normalized (``1``, ``2``, or ``3``)
        
        =================== ============================================ ===================
        Scalingmethod       Implementation                               Range
        =================== ============================================ ===================
        ``1``               As/Al* Tl/Ts                                 [0,1]
        ``2``               (As/Al - 1) / (Tl/Ts - 1)                    [-Ts/Tl,1]
        ``3``               Like ``2`` but clipping range at zero        [0,1]
        =================== ============================================ ===================
        
        '''
    
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
        
        if scalingmethod not in (1,2,3):
            raise Exception('Invalid argument to scalingrange argument.')

        if scalingmethod == 1:
            self.ydata = mavg_short/mavg_long * float(nshort)/float(nlong)
        elif scalingmethod in (2,3):
            self.ydata = (mavg_short/mavg_long - 1.) / ((float(nlong)/float(nshort)) - 1)
        
        if scalingmethod == 3:
            self.ydata = num.maximum(self.ydata, 0.)

    def peaks(self, threshold, tsearch, deadtime=False, nblock_duration_detection=100):
        '''Detect peaks above given threshold.
        
        From every instant, where the signal rises above *threshold*, a of time
        length of *tsearch* seconds is searched for a maximum. A list with
        tuples (time, value) for each detected peak is returned. The *deadtime*
        argument turns on a special deadtime duration detection algorithm useful
        in combination with recursive STA/LTA filters.
        
        '''
        y = self.ydata
        above =  num.where(y > threshold, 1, 0)
        deriv = num.zeros(y.size, dtype=num.int8)
        deriv[1:] = above[1:]-above[:-1]
        itrig_positions = num.nonzero(deriv>0)[0]
        tpeaks = []
        apeaks = []
        tzeros = []
        tzero = self.tmin
        
        for itrig_pos in itrig_positions:
            ibeg = itrig_pos
            iend = min(len(self.ydata), itrig_pos + tsearch/self.deltat)
            ipeak = num.argmax(y[ibeg:iend])
            tpeak = self.tmin + (ipeak+ibeg)*self.deltat
            apeak = y[ibeg+ipeak]

            if tpeak < tzero:
                continue

            if deadtime:
                ibeg = itrig_pos
                iblock = 0
                nblock = nblock_duration_detection
                totalsum = 0. 
                while True:
                    if ibeg+iblock*nblock >= len(y):
                        tzero = self.tmin + (len(y)-1)* self.deltat
                        break

                    logy = num.log(y[ibeg+iblock*nblock:ibeg+(iblock+1)*nblock])
                    logy[0] += totalsum
                    ysum = num.cumsum(logy)
                    totalsum = ysum[-1]
                    below = num.where(ysum <= 0., 1, 0)
                    deriv = num.zeros(ysum.size, dtype=num.int8)
                    deriv[1:] = below[1:]-below[:-1]
                    izero_positions = num.nonzero(deriv>0)[0] + iblock*nblock
                    if len(izero_positions) > 0:
                        tzero = self.tmin + (ibeg + izero_positions[0])*self.deltat
                        break
                    iblock += 1
            else:
                tzero = ibeg*self.deltat + self.tmin + tsearch

            tpeaks.append(tpeak)
            apeaks.append(apeak)
            tzeros.append(tzero)
        
        if deadtime:
            return tpeaks, apeaks, tzeros
        else:
            return tpeaks, apeaks

    def extend(self, tmin, tmax, fillmethod='zeros'):
        '''Extend trace to given span.
        
        :param tmin,tmax:  new span
        :param fillmethod: 'zeros' or 'repeat' 
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
        
        self._update_ids()
     
    def transfer(self, tfade, freqlimits, transfer_function=None, cut_off_fading=True):
        '''Return new trace with transfer function applied.
        
        :param tfade:             rise/fall time in seconds of taper applied in timedomain at both ends of trace.
        :param freqlimits:        4-tuple with corner frequencies in Hz.
        :param transfer_function: FrequencyResponse object; must provide a method 'evaluate(freqs)', which returns the
                                  transfer function coefficients at the frequencies 'freqs'.
        :param cut_off_fading:    whether to cut off rise/fall interval in output trace.
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
        '''Get FFT spectrum of trace.

        :param pad_to_pow2: whether to zero-pad the data to next larger power-of-two length
        :param tfade: ``None`` or a time length in seconds, to apply cosine shaped tapers to both

        :returns: a tuple with (frequencies, values)
        '''
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

    def multi_filter(self, filter_freqs, bandwidth): 

        class Gauss(FrequencyResponse):
            def __init__(self, f0, a=1.0):
                self._omega0 = 2.*math.pi*f0
                self._a = a

            def evaluate(self, freqs):
                omega = 2.*math.pi*freqs
                return num.exp(-((omega-self._omega0)/(self._a*self._omega0))**2)

        freqs, coefs = self.spectrum()
        y = self.get_ydata()
        trs = []
        n = self.data_len()
        nfilt = len(filter_freqs)
        signal_tf = num.zeros((nfilt, n))
        centroid_freqs = num.zeros(nfilt)
        for ifilt, f0 in enumerate(filter_freqs):
            taper = Gauss(f0, a=bandwidth)
            weights = taper.evaluate(freqs)
            nhalf = freqs.size
            analytic_spec = num.zeros(n, dtype=num.complex)
            analytic_spec[:nhalf] = coefs*weights

            enorm = num.abs(analytic_spec[:nhalf])**2
            enorm /= num.sum(enorm)

            if n % 2 == 0:
                analytic_spec[1:nhalf-1] *= 2.
            else:
                analytic_spec[1:nhalf] *= 2.

            analytic = num.fft.ifft(analytic_spec)
            signal_tf[ifilt,:] = num.abs(analytic)

            enorm = num.abs(analytic_spec[:nhalf])**2
            enorm /= num.sum(enorm)
            centroid_freqs[ifilt] = num.sum(freqs*enorm)

        return centroid_freqs, signal_tf
        
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
        '''Fill string template with trace metadata.
        
        Uses normal pyton '%(placeholder)s' string templates. The following
        placeholders are considered: ``network``, ``station``, ``location``,
        ``channel``, ``tmin`` (time of first sample), ``tmax`` (time of last
        sample), ``tmin_ms``, ``tmax_ms``, ``tmin_us``, ``tmax_us``. The
        versions with '_ms' include milliseconds, the versions with '_us'
        include microseconds.
        '''

        params = dict(zip( ('network', 'station', 'location', 'channel'), self.nslc_id))
        params['tmin'] = util.time_to_str(self.tmin, format='%Y-%m-%d_%H-%M-%S')
        params['tmax'] = util.time_to_str(self.tmax, format='%Y-%m-%d_%H-%M-%S')
        params['tmin_ms'] = util.time_to_str(self.tmin, format='%Y-%m-%d_%H-%M-%S.3FRAC')
        params['tmax_ms'] = util.time_to_str(self.tmax, format='%Y-%m-%d_%H-%M-%S.3FRAC')
        params['tmin_us'] = util.time_to_str(self.tmin, format='%Y-%m-%d_%H-%M-%S.6FRAC')
        params['tmax_us'] = util.time_to_str(self.tmax, format='%Y-%m-%d_%H-%M-%S.6FRAC')
        params.update(additional)
        return template % params

    def plot(self):
        '''Show trace with matplotlib.
        
        See also: :py:meth:`Trace.snuffle`.
        '''
        import pylab
        pylab.plot(self.get_xdata(), self.get_ydata())
        name = self.channel+' '+self.station+' '+time.strftime("%d-%m-%y %H:%M:%S", time.gmtime(self.tmin))+' - '+time.strftime("%d-%m-%y %H:%M:%S", time.gmtime(self.tmax))
        pylab.title(name)
        pylab.show()
    
    def snuffle(self, **kwargs):
        '''Show trace in a snuffler window.

        :param stations: list of `pyrocko.model.Station` objects or ``None``
        :param events: list of `pyrocko.model.Event` objects or ``None``
        :param markers: list of `pyrocko.gui_util.Marker` objects or ``None``
        :param ntracks: float, number of tracks to be shown initially (default: 12)
        :param follow: time interval (in seconds) for real time follow mode or ``None``
        :param controls: bool, whether to show the main controls (default: ``True``)
        :param opengl: bool, whether to use opengl (default: ``False``)
        '''

        return snuffle( [self], **kwargs)

def snuffle(traces, **kwargs):
    '''Show traces in a snuffler window.

    :param stations: list of `pyrocko.model.Station` objects or ``None``
    :param events: list of `pyrocko.model.Event` objects or ``None``
    :param markers: list of `pyrocko.gui_util.Marker` objects or ``None``
    :param ntracks: float, number of tracks to be shown initially (default: 12)
    :param follow: time interval (in seconds) for real time follow mode or ``None``
    :param controls: bool, whether to show the main controls (default: ``True``)
    :param opengl: bool, whether to use opengl (default: ``False``)
    '''

    from pyrocko import pile, snuffler
    p = pile.Pile()
    if traces:
        trf = pile.MemTracesFile(p, traces)
        p.add_file(trf)
    return snuffler.snuffle(p, **kwargs)

class NoData(Exception):
    '''This exception is raised by some :py:class:`Trace` operations when no or not enough data is available.'''
    pass

class AboveNyquist(Exception):
    '''This exception is raised by some :py:class:`Trace` operations when given frequencies are above the Nyquist frequency.'''
    pass

class TraceTooShort(Exception):
    '''This exception is raised by some :py:class:`Trace` operations when the trace is too short.'''
    pass

class ResamplingFailed(Exception):
    pass

def minmax(traces, key=None, mode='minmax'):
    
    '''Get data range given traces grouped by selected pattern.
   
    :param key: a callable which takes as single argument a trace and returns a key for the grouping of the results.
                If this is ``None``, the default, ``lambda tr: (tr.network, tr.station, tr.location, tr.channel)`` is used.
    :param mode: 'minmax' or floating point number. If this is 'minmax', minimum and maximum of the traces are used, 
                 if it is a number, mean +- stdandard deviation times *mode* is used.
    
    :returns: a dict with the combined data ranges.

    Examples::
    
        ranges = minmax(traces, lambda tr: tr.channel)
        print ranges['N']   # print minimum and maximum of all traces with channel == 'N'
        print ranges['E']   # print mimimum and maximum of all traces with channel == 'E'

        ranges = minmax(traces, lambda tr: (tr.network, tr.station))
        print ranges['GR', 'HAM3']    # print minmum and maxium of all traces with 
                                      # network == 'GR' and station == 'HAM3'

        ranges = minmax(traces, lambda tr: None)
        print ranges[None]  # prints minimum and maximum of all traces


    '''
   
    if key is None:
        key = _default_key

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
        
def minmaxtime(traces, key=None):
    
    '''Get time range given traces grouped by selected pattern.
    
    :param key: a callable which takes as single argument a trace and returns a key for the grouping of the results.
                If this is ``None``, the default, ``lambda tr: (tr.network, tr.station, tr.location, tr.channel)`` is used.
    
    :returns: a dict with the combined data ranges.
    '''
    
    if key is None:
        key = _default_key

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
    
def degapper(traces, maxgap=5, fillmethod='interpolate', deoverlap='use_second', maxlap=None):
    
    '''Try to connect traces and remove gaps.
    
    This method will combine adjacent traces, which match in their network, 
    station, location and channel attributes. Overlapping parts are handled
    according to the `deoverlap` argument.
    
    :param traces:      input traces, must be sorted by their full_id attribute.
    :param maxgap:      maximum number of samples to interpolate.
    :param fillmethod:  what to put into the gaps: 'interpolate' or 'zeros'.
    :param deoverlap:   how to handle overlaps: 'use_second' to use data from 
                        second trace (default), 'use_first' to use data from first
                        trace, 'crossfade_cos' to crossfade with cosine taper 
    :param maxlap:      maximum number of samples of overlap which are removed
      
    :returns:           list of traces
    '''

    in_traces = traces 
    out_traces = []
    if not in_traces: return out_traces
    out_traces.append(in_traces.pop(0))
    while in_traces:
        
        a = out_traces[-1]
        b = in_traces.pop(0)
        
        avirt, bvirt = a.ydata is None, b.ydata is None
        assert avirt == bvirt, 'traces given to degapper() must either all have data or have no data.'
        virtual = avirt and bvirt

        if (a.nslc_id == b.nslc_id and a.deltat == b.deltat and 
            a.data_len() >= 1 and b.data_len() >= 1 and 
            (virtual or a.ydata.dtype == b.ydata.dtype)):
            
            dist = (b.tmin-(a.tmin+(a.data_len()-1)*a.deltat))/a.deltat
            idist = int(round(dist))
            if abs(dist - idist) > 0.05 and idist <= maxgap:
                pass #logger.warn('Cannot degap traces with displaced sampling (%s,%s,%s,%s)' % a.nslc_id)
            else:
                if 1 < idist <= maxgap:
                    if not virtual:
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
                    if not virtual:
                        a.ydata = num.concatenate((a.ydata,b.ydata))
                    a.tmax = b.tmax
                    if a.mtime and b.mtime:
                        a.mtime = max(a.mtime, b.mtime)
                    continue
                    
                elif idist <= 0 and (maxlap is None or -maxlap < idist):
                    if b.tmax > a.tmax:
                        if not virtual:
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
                    
        if b.data_len() >= 1:
            out_traces.append(b)
            
    for tr in out_traces:
        tr._update_ids()
    
    return out_traces

def rotate(traces, azimuth, in_channels, out_channels):
    '''2D rotation of traces.
    
    :param traces: list of input traces
    :param azimuth: difference of the azimuths of the component directions
                     (azimuth of out_channels[0]) - (azimuth of in_channels[0])
    :param in_channels: names of the input channels (e.g. 'N', 'E')
    :param out_channels: names of the output channels (e.g. 'R', 'T')
    :returns: list of rotated traces 
    '''
    
    phi = azimuth/180.*math.pi
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    rotated = []
    in_channels = tuple(_channels_to_names(in_channels))
    out_channels = tuple(_channels_to_names(out_channels))
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

def rotate_to_rt(n, e, source, receiver, out_channels=('R', 'T')):
    azimuth = orthodrome.azimuth(receiver, source) + 180.
    in_channels = n.channel, e.channel
    out = rotate([n,e], azimuth, in_channels=in_channels, out_channels=out_channels)
    assert len(out) == 2
    for tr in out:
        if tr.channel=='R':
            r = tr
        elif tr.channel == 'T':
            t = tr

    return r,t

def _decompose(a):
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

def _channels_to_names(channels):
    names = []
    for ch in channels:
        if isinstance(ch, model.Channel):
            names.append(ch.name)
        else:
            names.append(ch)
    return names

def project(traces, matrix, in_channels, out_channels):
   
    '''Affine transform of three-component traces.

    Compute matrix-vector product of three-component traces, to e.g. rotate
    traces into a different basis. The traces are distinguished and ordered by their channel attribute.
    The tranform is applied to overlapping parts of any appropriate combinations of the input traces. This
    should allow this function to be robust with data gaps.
    It also tries to apply the tranformation
    to subsets of the channels, if this is possible, so that, if for example a vertical
    compontent is missing, horizontal components can still be rotated.

    :param traces: list of traces in arbitrary order
    :param matrix: tranformation matrix
    :param in_channels: input channel names
    :param out_channels: output channel names
    :returns: list of transformed traces
    '''
    
    in_channels = tuple( _channels_to_names(in_channels) )
    out_channels = tuple( _channels_to_names(out_channels) )
    systems = _decompose(matrix)
    
    # fallback to full matrix if some are not quadratic
    for iins, iouts, submatrix in systems:
        if submatrix.shape[0] != submatrix.shape[1]:
            return _project3(traces, matrix, in_channels, out_channels)
    
    projected = []
    for iins, iouts ,submatrix in systems:
        in_cha = tuple( [ in_channels[iin] for iin in iins ] )
        out_cha = tuple( [ out_channels[iout] for iout in iouts ] )
        if submatrix.shape[0] == 1:
            projected.extend( _project1(traces, submatrix, in_cha, out_cha) )
        elif submatrix.shape[1] == 2:
            projected.extend( _project2(traces, submatrix, in_cha, out_cha) )
        else:
            projected.extend( _project3(traces, submatrix, in_cha, out_cha) )
    
   
    return projected

def project_dependencies(matrix, in_channels, out_channels):
    
    '''Figure out what dependencies project() would produce.'''
    
    in_channels = tuple( _channels_to_names(in_channels) )
    out_channels = tuple( _channels_to_names(out_channels) )
    systems = _decompose(matrix)
    
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
        
def _project1(traces, matrix, in_channels, out_channels):
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

def _project2(traces, matrix, in_channels, out_channels):
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
            
            if tmin > tmax:
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
            
def _project3(traces, matrix, in_channels, out_channels):
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


def correlate(a, b, mode='valid', normalization=None, use_fft=False):
    '''Cross correlation of two traces.
    
    :param a,b: input traces
    :param mode: ``'valid'``, ``'full'``, or ``'same'``
    :param normalization: ``'normal'``, ``'gliding'``, or ``None``
    :param use_fft: bool, whether to do cross correlation in spectral domain

    :returns: trace containing cross correlation coefficients

    This function computes the cross correlation between two traces. It
    evaluates the discrete equivalent of

    .. math::

       c(t) = \\int_{-\\infty}^{\\infty} a^{\\ast}(\\tau) b(t+\\tau) d\\tau

    where the star denotes complex conjugate. Note, that the arguments here are
    swapped when compared with the :py:func:`numpy.correlate` function,
    which is internally called. This function should be safe even with older
    versions of NumPy, where the correlate function has some problems.

    A trace containing the cross correlation coefficients is returned. The time
    information of the output trace is set so that the returned cross
    correlation can be viewed directly as a function of time lag. 

    Example::
        
        # align two traces a and b containing a time shifted similar signal:
        c = pyrocko.trace.correlate(a,b)
        t, coef = c.max()  # get time and value of maximum
        b.shift(-t)        # align b with a

    '''

    assert_same_sampling_rate(a,b)

    ya, yb = a.ydata, b.ydata

    yc = numpy_correlate_fixed(yb, ya, mode=mode, use_fft=use_fft) # need reversed order here
    kmin, kmax = numpy_correlate_lag_range(yb, ya, mode=mode, use_fft=use_fft)

    if normalization == 'normal':
        normfac = num.sqrt(num.sum(ya**2))*num.sqrt(num.sum(yb**2))
        yc /= normfac

    elif normalization == 'gliding':
        if mode != 'valid':
            assert False, 'gliding normalization currently only available with "valid" mode.'

        if ya.size < yb.size:
            yshort, ylong = ya, yb
        else:
            yshort, ylong = yb, ya
        
        epsilon = 0.00001
        normfac_short = num.sqrt(num.sum(yshort**2)) 
        normfac = normfac_short * num.sqrt(moving_sum(ylong**2,yshort.size, mode='valid')) + normfac_short*epsilon
        
        if yb.size <= ya.size:
            normfac = normfac[::-1]

        yc /= normfac
    
    c = a.copy()
    c.set_ydata(yc)
    c.set_codes(*merge_codes(a,b,'~'))
    c.shift(-c.tmin + b.tmin-a.tmin + kmin * c.deltat)

    return c

def deconvolve(a, b, waterlevel, tshift=0., pad=0.5, fd_taper=None, pad_to_pow2=True):
    
    same_sampling_rate(a,b)
    assert abs(a.tmin - b.tmin) < a.deltat * 0.001
    deltat = a.deltat
    npad = int(round(a.data_len()*pad + tshift / deltat))
    
    ndata = max(a.data_len(), b.data_len())
    ndata_pad = ndata + npad
    
    if pad_to_pow2:
        ntrans = nextpow2(ndata_pad)
    else:
        ntrans = ndata
    
    aspec = num.fft.rfft(a.ydata, ntrans)
    bspec = num.fft.rfft(b.ydata, ntrans)

    out = aspec * num.conj(bspec)

    bautocorr = bspec*num.conj(bspec)
    denom = num.maximum( bautocorr, waterlevel * bautocorr.max() )
    
    out /= denom
    df = 1/(ntrans*deltat)

    if fd_taper is not None:
        fd_taper( out, 0.0, df )

    ydata = num.roll(num.fft.irfft(out),int(round(tshift/deltat)))
    c = a.copy(data=False)
    c.set_ydata(ydata[:ndata])
    c.set_codes(*merge_codes(a,b,'/'))
    return c

def assert_same_sampling_rate(a,b, eps=1.0e-6):
    assert same_sampling_rate(a,b,eps), 'Sampling rates differ: %g != %g' % (a.deltat, b.deltat)

def same_sampling_rate(a,b, eps=1.0e-6):
    '''Check if two traces have the same sampling rate.
    
    :param a,b: input traces
    :param eps: relative tolerance
    '''
    return abs(a.deltat - b.deltat) < (a.deltat + b.deltat)*eps

def merge_codes(a,b, sep='-'):
    '''Merge network-station-location-channel codes of a pair of traces.'''
    
    
    o = []
    for xa,xb in zip(a.nslc_id, b.nslc_id):
        if xa == xb:
            o.append(xa)
        else:
            o.append(sep.join((xa,xb)))
    return o

class Taper(object):

    def __call__(self, y, x0, dx):
        pass

class CosTaper(Taper):
    def __init__(self, a,b,c,d):
        self._corners = (a,b,c,d)

    def __call__(self, y, x0, dx):
        a,b,c,d = self._corners
        apply_costaper(a, b, c, d, y, x0, dx)

class CosFader(Taper):
    def __init__(self, xfade=None, xfrac=None):
        assert (xfade is None) != (xfrac is None)
        self._xfade = xfade
        self._xfrac = xfrac
    
    def __call__(self, y, x0, dx):
        xfade = self._xfade

        xlen = (y.size - 1)*dx
        if xfade is None:
            xfade = xlen * xfrac

        a = x0
        b = x0 + self._xfade
        c = x0 + xlen - self._xfade
        d = x0 + xlen

        apply_costaper(a, b, c, d, y, x0, dx)

class GaussTaper(Taper):

    def __init__(self, alpha):
        self._alpha = alpha

    def __call__(self, y, x0, dx):
        f = x0 + num.arange( y.size )*dx
        y *= num.exp(-(2.*num.pi)**2/(4.*self._alpha**2) * f**2)

class FrequencyResponse(object):
    '''Evaluates frequency response at given frequencies.'''
    
    def evaluate(self, freqs):
        coefs = num.ones(freqs.size, dtype=num.complex)
        return coefs
   
class InverseEvalresp(FrequencyResponse):
    '''Calls evalresp and generates values of the inverse instrument response for 
       deconvolution of instrument response.
       
       :param respfile: response file in evalresp format
       :param trace: trace for which the response is to be extracted from the file
       :param target: ``'dis'`` for displacement or ``'vel'`` for velocity
       '''
    
    def __init__(self, respfile, trace, target='dis'):
        self.respfile = respfile
        self.nslc_id = trace.nslc_id
        self.instant = (trace.tmin + trace.tmax)/2.
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
    '''Evaluates frequency response from pole-zero representation.

    ::

                           (j*2*pi*f - zeros[0]) * (j*2*pi*f - zeros[1]) * ... 
         T(f) = constant * -------------------------------------------------------
                           (j*2*pi*f - poles[0]) * (j*2*pi*f - poles[1]) * ...
    
   
    The poles and zeros should be given as angular frequencies, not in Hz.
    '''
    
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
    '''Interpolates frequency response given at a set of sampled frequencies.
    
    :param freqs,vals: frequencies and values of the sampled response function.
    :param left,right: values to return when input is out of range. If set to ``None`` (the default) the endpoints are returned.
    '''
    
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
        '''Get inverse as a new :py:class:`SampledResponse` object.'''

        def inv_or_none(x):
            if x is not None:
                return 1./x
            
        return SampledResponse(self.freqs, 1./self.vals, left=inv_or_none(self.left), right=inv_or_none(self.right))
    
    def frequencies(self):
        return self.freqs
    
    def values(self):
        return self.vals
    
class IntegrationResponse(FrequencyResponse):
    '''The integration response, optionally multiplied by a constant gain.

    ::

                    gain
        T(f) = --------------
               (j*2*pi * f)^n
    '''

    def __init__(self, n=1, gain=1.0):
        self._n = n
        self._gain = gain
        
    def evaluate(self, freqs):
        return self._gain / (1.0j * 2. * num.pi*freqs)**self._n

class DifferentiationResponse(FrequencyResponse):
    '''The differentiation response, optionally multiplied by a constant gain.

    ::

        T(f) = gain * (j*2*pi * f)^n
    '''

    def __init__(self, n=1, gain=1.0):
        self._n = n
        self._gain = gain
        
    def evaluate(self, freqs):
        return self._gain * (1.0j * 2. * num.pi * freqs)**self._n

class AnalogFilterResponse(FrequencyResponse):
    '''Frequency response of an analog filter.
    
    (see :py:func:`scipy.signal.freqs`).'''

    def __init__(self, b,a):
        self._b = b
        self._a = a
    
    def evaluate(self, freqs):
        return signal.freqs(self._b, self._a, freqs/(2.*pi))[1]

class MultiplyResponse(FrequencyResponse):
    '''Multiplication of two :py:class:`FrequencyResponse` objects.'''

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def evaluate(self, freqs):
        return self._a.evaluate(freqs) * self._b.evaluate(freqs)

if sys.version_info >= (2,5):
    from need_python_2_5.trace import *

cached_coefficients = {}
def _get_cached_filter_coefs(order, corners, btype):
    ck = (order, tuple(corners), btype)
    if ck not in cached_coefficients:
        if len(corners) == 0:
            cached_coefficients[ck] = signal.butter(order, corners[0], btype=btype)
        else:
            cached_coefficients[ck] = signal.butter(order, corners, btype=btype)

    return cached_coefficients[ck]
    
    
class _globals:
    _numpy_has_correlate_flip_bug = None

_default_key = lambda tr: (tr.network, tr.station, tr.location, tr.channel)

def numpy_has_correlate_flip_bug():
    '''Check if NumPy's correlate function reveals old behaviour'''

    if _globals._numpy_has_correlate_flip_bug is None:
        a = num.array([0,0,1,0,0,0,0])
        b = num.array([0,0,0,0,1,0,0,0])
        ab = num.correlate(a,b, mode='same')
        ba = num.correlate(b,a, mode='same')
        _globals._numpy_has_correlate_flip_bug = num.all(ab == ba)
    
    return _globals._numpy_has_correlate_flip_bug

def numpy_correlate_fixed(a,b, mode='valid', use_fft=False):
    '''Call :py:func:`numpy.correlate` with fixes.
   
        c[k] = sum_i a[i+k] * conj(b[i]) 

    Note that the result produced by newer numpy.correlate is always flipped
    with respect to the formula given in its documentation
    (if ascending k assumed for the output).
    '''

    if use_fft:
        return signal.fftconvolve(a, b[::-1], mode=mode)

    else:
        buggy = numpy_has_correlate_flip_bug()

        a = num.asarray(a)
        b = num.asarray(b)

        if buggy:
            b = num.conj(b)

        c = num.correlate(a,b,mode=mode)

        if buggy and a.size < b.size:
            return c[::-1]
        else:
            return c

def numpy_correlate_emulate(a,b, mode='valid'):
    '''Slow version of :py:func:`numpy.correlate` for comparison.'''

    a = num.asarray(a)
    b = num.asarray(b)
    kmin = -(b.size-1)
    klen = a.size-kmin
    kmin, kmax = numpy_correlate_lag_range(a,b, mode=mode)
    klen = kmax - kmin + 1
    c = num.zeros(klen, dtype=num.find_common_type((b.dtype, a.dtype), ()))
    for k in xrange(kmin,kmin+klen):
        imin = max(0, -k)
        ilen = min(b.size, a.size-k) - imin 
        c[k-kmin] = num.sum( a[imin+k:imin+ilen+k] * num.conj(b[imin:imin+ilen]) )

    return c

def numpy_correlate_lag_range(a,b, mode='valid', use_fft=False):
    '''Get range of lags for which :py:func:`numpy.correlate` produces values.'''

    a = num.asarray(a)
    b = num.asarray(b)

    kmin = -(b.size-1)
    if mode == 'full':
        klen = a.size-kmin
    elif mode == 'same': 
        klen = max(a.size, b.size)
        kmin += (a.size+b.size-1 - max(a.size, b.size))/2 + \
                int(not use_fft and a.size % 2 == 0 and b.size > a.size)
    elif mode == 'valid':
        klen = abs(a.size - b.size) + 1 
        kmin += min(a.size, b.size) - 1

    return kmin, kmin + klen - 1

def autocorr(x, nshifts):
    '''Compute biased estimate of the first autocorrelation coefficients.
    
    :param x: input array
    :param nshifts: number of coefficients to calculate
    '''

    mean = num.mean(x)
    std = num.std(x)
    n = x.size
    xdm = x - mean
    r = num.zeros(nshifts)
    for k in range(nshifts):
        r[k] = 1./((n-num.abs(k))*std) * num.sum( xdm[:n-k] * xdm[k:] )
        
    return r

def yulewalker(x, order):
    '''Compute autoregression coefficients using Yule-Walker method.
    
    :param x: input array
    :param order: number of coefficients to produce

    A biased estimate of the autocorrelation is used. The Yule-Walker equations
    are solved by :py:func:`numpy.linalg.inv` instead of Levinson-Durbin
    recursion which is normally used.
    '''

    gamma = autocorr(x, order+1)
    d = gamma[1:1+order]
    a = num.zeros((order,order))
    gamma2 = num.concatenate( (gamma[::-1], gamma[1:order]) )
    for i in range(order):
        ioff = order-i
        a[i,:] = gamma2[ioff:ioff+order]
    
    return num.dot(num.linalg.inv(a),-d)

def moving_avg(x,n):
    n = int(n)
    cx = x.cumsum()
    nn = len(x)
    y = num.zeros(nn, dtype=cx.dtype)
    y[n/2:n/2+(nn-n)] = (cx[n:]-cx[:-n])/n
    y[:n/2] = y[n/2]
    y[n/2+(nn-n):] = y[n/2+(nn-n)-1]
    return y

def moving_sum(x,n, mode='valid'):
    n = int(n)
    cx = x.cumsum()
    nn = len(x)

    if mode == 'valid':
        if nn-n+1 <= 0:
            return num.zeros(0, dtype=cx.dtype) 
        y = num.zeros(nn-n+1, dtype=cx.dtype)
        y[0] = cx[n-1]
        y[1:nn-n+1] = cx[n:nn]-cx[0:nn-n]
    
    if mode == 'full':
        y = num.zeros(nn+n-1, dtype=cx.dtype)
        if n <= nn:
            y[0:n] = cx[0:n]
            y[n:nn] = cx[n:nn]-cx[0:nn-n]
            y[nn:nn+n-1] = cx[-1]-cx[nn-n:nn-1]
        else:
            y[0:nn] = cx[0:nn]
            y[nn:n] = cx[nn-1]
            y[n:nn+n-1] = cx[nn-1] - cx[0:nn-1]

    if mode == 'same':
        n1 = (n-1)/2
        y = num.zeros(nn, dtype=cx.dtype)
        if n <= nn:
            y[0:n-n1] = cx[n1:n]
            y[n-n1:nn-n1] = cx[n:nn]-cx[0:nn-n]
            y[nn-n1:nn] = cx[nn-1] - cx[nn-n:nn-n+n1]
        else:
            y[0:max(0,nn-n1)] = cx[min(n1,nn):nn]
            y[max(nn-n1,0):min(n-n1,nn)] = cx[nn-1]
            y[min(n-n1,nn):nn] = cx[nn-1] - cx[0:max(0,nn-(n-n1))]


    return y

def nextpow2(i):
    return 2**int(math.ceil(math.log(i)/math.log(2.)))
    
def snapper_w_offset(nmax, offset, delta, snapfun=math.ceil):
    def snap(x):
        return max(0,min(snapfun((x-offset)/delta),nmax))
    return snap

def snapper(nmax, delta, snapfun=math.ceil):
    def snap(x):
        return max(0,min(snapfun(x/delta),nmax))
    return snap

def apply_costaper(a, b, c, d, y, x0, dx):
    hi = snapper_w_offset(y.size, x0, dx)
    y[:hi(a)] = 0.
    y[hi(a):hi(b)] *= 0.5 - 0.5*num.cos((dx*num.arange(hi(a),hi(b))-(a-x0))/(b-a)*num.pi)
    y[hi(c):hi(d)] *= 0.5 + 0.5*num.cos((dx*num.arange(hi(c),hi(d))-(c-x0))/(d-c)*num.pi)
    y[hi(d):] = 0.

def costaper(a,b,c,d, nfreqs, deltaf):
    hi = snapper(nfreqs, deltaf)
    tap = num.zeros(nfreqs)
    tap[hi(a):hi(b)] = 0.5 - 0.5*num.cos((deltaf*num.arange(hi(a),hi(b))-a)/(b-a)*num.pi)
    tap[hi(b):hi(c)] = 1.
    tap[hi(c):hi(d)] = 0.5 + 0.5*num.cos((deltaf*num.arange(hi(c),hi(d))-c)/(d-c)*num.pi)
    
    return tap

def t2ind(t,tdelta, snap=round):
    return int(snap(t/tdelta))

def hilbert(x, N=None):
    '''Return the hilbert transform of x of length N.

    (from scipy.signal, but changed to use fft and ifft from numpy.fft)
    '''
    x = num.asarray(x)
    if N is None:
        N = len(x)
    if N <=0:
        raise ValueError, "N must be positive."
    if num.iscomplexobj(x):
        print "Warning: imaginary part of x ignored."
        x = real(x)
    Xf = num.fft.fft(x,N,axis=0)
    h = num.zeros(N)
    if N % 2 == 0:
        h[0] = h[N/2] = 1
        h[1:N/2] = 2
    else:
        h[0] = 1
        h[1:(N+1)/2] = 2

    if len(x.shape) > 1:
        h = h[:, newaxis]
    x = num.fft.ifft(Xf*h)
    return x
    
        
