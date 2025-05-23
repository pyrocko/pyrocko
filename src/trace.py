# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Basic signal processing for seismic waveforms.
'''

import time
import math
import copy
import logging
import hashlib
import re
from collections import defaultdict

import numpy as num
from scipy import signal

from pyrocko import util, orthodrome, pchain, model, signal_ext
from pyrocko.util import reuse
from pyrocko.guts import Object, Float, Int, String, List, \
    StringChoice, Timestamp
from pyrocko.guts_array import Array
from pyrocko.model import squirrel_content

# backward compatibility
from pyrocko.util import UnavailableDecimation  # noqa
from pyrocko.response import (  # noqa
    FrequencyResponse, Evalresp, InverseEvalresp, PoleZeroResponse,
    ButterworthResponse, SampledResponse, IntegrationResponse,
    DifferentiationResponse, MultiplyResponse)


guts_prefix = 'pf'

logger = logging.getLogger('pyrocko.trace')


g_tapered_coeffs_cache = {}
g_one_response = FrequencyResponse()


@squirrel_content
class Trace(Object):

    '''
    Create new trace object.

    A ``Trace`` object represents a single continuous strip of evenly sampled
    time series data.  It is built from a 1D NumPy array containing the data
    samples and some attributes describing its beginning  and ending time, its
    sampling rate and four string identifiers (its network, station, location
    and channel code).

    :param network: network code
    :param station: station code
    :param location: location code
    :param channel: channel code
    :param extra: extra code
    :param tmin: system time of first sample in [s]
    :param tmax: system time of last sample in [s] (if set to ``None`` it is
        computed from length of ``ydata``)
    :param deltat: sampling interval in [s]
    :param ydata: 1D numpy array with data samples (can be ``None`` when
        ``tmax`` is not ``None``)
    :param mtime: optional modification time

    :param meta: additional meta information (not used, but maintained by the
        library)

    The length of the network, station, location and channel codes is not
    resricted by this software, but data formats like SAC, Mini-SEED or GSE
    have different limits on the lengths of these codes. The codes set here are
    silently truncated when the trace is stored
    '''

    network = String.T(default='', help='Deployment/network code (1-8)')
    station = String.T(default='STA', help='Station code (1-5)')
    location = String.T(default='', help='Location code (0-2)')
    channel = String.T(default='', help='Channel code (3)')
    extra = String.T(default='', help='Extra/custom code')

    tmin = Timestamp.T(default=Timestamp.D('1970-01-01 00:00:00'))
    tmax = Timestamp.T()

    deltat = Float.T(default=1.0)
    ydata = Array.T(optional=True, shape=(None,), serialize_as='base64+meta')

    mtime = Timestamp.T(optional=True)

    cached_frequencies = {}

    def __init__(self, network='', station='STA', location='', channel='',
                 tmin=0., tmax=None, deltat=1., ydata=None, mtime=None,
                 meta=None, extra=''):

        Object.__init__(self, init_props=False)

        time_float = util.get_time_float()

        if not isinstance(tmin, time_float):
            tmin = Trace.tmin.regularize_extra(tmin)

        if tmax is not None and not isinstance(tmax, time_float):
            tmax = Trace.tmax.regularize_extra(tmax)

        if mtime is not None and not isinstance(mtime, time_float):
            mtime = Trace.mtime.regularize_extra(mtime)

        if ydata is not None and not isinstance(ydata, num.ndarray):
            ydata = Trace.ydata.regularize_extra(ydata)

        self._growbuffer = None

        tmin = time_float(tmin)
        if tmax is not None:
            tmax = time_float(tmax)

        if mtime is None:
            mtime = time_float(time.time())

        self.network, self.station, self.location, self.channel, \
            self.extra = [
                reuse(x) for x in (
                    network, station, location, channel, extra)]

        self.tmin = tmin
        self.deltat = deltat

        if tmax is None:
            if ydata is not None:
                self.tmax = self.tmin + (ydata.size-1)*self.deltat
            else:
                raise Exception(
                    'fixme: trace must be created with tmax or ydata')
        else:
            n = int(round((tmax - self.tmin) / self.deltat)) + 1
            self.tmax = self.tmin + (n - 1) * self.deltat

        self.meta = meta
        self.ydata = ydata
        self.mtime = mtime
        self._update_ids()
        self.file = None
        self._pchain = None

    def __str__(self):
        fmt = min(9, max(0, -int(math.floor(math.log10(self.deltat)))))
        s = 'Trace (%s, %s, %s, %s)\n' % self.nslc_id
        s += '  timerange: %s - %s\n' % (
            util.time_to_str(self.tmin, format=fmt),
            util.time_to_str(self.tmax, format=fmt))

        s += '  delta t: %g\n' % self.deltat
        if self.meta:
            for k in sorted(self.meta.keys()):
                s += '  %s: %s\n' % (k, self.meta[k])
        return s

    @property
    def codes(self):
        from pyrocko.squirrel import model
        return model.CodesNSLCE(
            self.network, self.station, self.location, self.channel,
            self.extra)

    @property
    def time_span(self):
        return self.tmin, self.tmax

    @property
    def summary_entries(self):
        return (
            self.__class__.__name__,
            str(self.codes),
            self.str_time_span,
            '%g' % (1.0/self.deltat))

    @property
    def summary_stats_entries(self):
        return tuple('%13.7g' % v for v in (
            self.ydata.min(),
            self.ydata.max(),
            self.ydata.mean(),
            self.ydata.std()))

    @property
    def summary(self):
        return util.fmt_summary(self.summary_entries, (10, 20, 55, 0))

    @property
    def summary_stats(self):
        return self.summary + ' | ' + util.fmt_summary(
            self.summary_stats_entries, (12, 12, 12, 12))

    def __getstate__(self):
        return (self.network, self.station, self.location, self.channel,
                self.tmin, self.tmax, self.deltat, self.mtime,
                self.ydata, self.meta, self.extra)

    def __setstate__(self, state):
        if len(state) == 11:
            self.network, self.station, self.location, self.channel, \
                self.tmin, self.tmax, self.deltat, self.mtime, \
                self.ydata, self.meta, self.extra = state

        elif len(state) == 12:
            # backward compatibility 0
            self.network, self.station, self.location, self.channel, \
                self.tmin, self.tmax, self.deltat, self.mtime, \
                self.ydata, self.meta, _, self.extra = state

        elif len(state) == 10:
            # backward compatibility 1
            self.network, self.station, self.location, self.channel, \
                self.tmin, self.tmax, self.deltat, self.mtime, \
                self.ydata, self.meta = state

            self.extra = ''

        else:
            # backward compatibility 2
            self.network, self.station, self.location, self.channel, \
                self.tmin, self.tmax, self.deltat, self.mtime = state
            self.ydata = None
            self.meta = None
            self.extra = ''

        self._growbuffer = None
        self._update_ids()

    def hash(self, unsafe=False):
        sha1 = hashlib.sha1()
        for code in self.nslc_id:
            sha1.update(code.encode())
        sha1.update(self.tmin.hex().encode())
        sha1.update(self.tmax.hex().encode())
        sha1.update(self.deltat.hex().encode())

        if self.ydata is not None:
            if not unsafe:
                sha1.update(memoryview(self.ydata))
            else:
                sha1.update(memoryview(self.ydata[:32]))
                sha1.update(memoryview(self.ydata[-32:]))

        return sha1.hexdigest()

    def name(self):
        '''
        Get a short string description.
        '''

        s = '%s.%s.%s.%s, %s, %s' % (self.nslc_id + (
            util.time_to_str(self.tmin),
            util.time_to_str(self.tmax)))

        return s

    def __eq__(self, other):
        return (
            isinstance(other, Trace)
            and self.network == other.network
            and self.station == other.station
            and self.location == other.location
            and self.channel == other.channel
            and (abs(self.deltat - other.deltat)
                 < (self.deltat + other.deltat)*1e-6)
            and abs(self.tmin-other.tmin) < self.deltat*0.01
            and abs(self.tmax-other.tmax) < self.deltat*0.01
            and num.all(self.ydata == other.ydata))

    def almost_equal(self, other, rtol=1e-5, atol=0.0):
        return (
            self.network == other.network
            and self.station == other.station
            and self.location == other.location
            and self.channel == other.channel
            and (abs(self.deltat - other.deltat)
                 < (self.deltat + other.deltat)*1e-6)
            and abs(self.tmin-other.tmin) < self.deltat*0.01
            and abs(self.tmax-other.tmax) < self.deltat*0.01
            and num.allclose(self.ydata, other.ydata, rtol=rtol, atol=atol))

    def assert_almost_equal(self, other, rtol=1e-5, atol=0.0):

        assert self.network == other.network, \
            'network codes differ: %s, %s' % (self.network, other.network)
        assert self.station == other.station, \
            'station codes differ: %s, %s' % (self.station, other.station)
        assert self.location == other.location, \
            'location codes differ: %s, %s' % (self.location, other.location)
        assert self.channel == other.channel, 'channel codes differ'
        assert (abs(self.deltat - other.deltat)
                < (self.deltat + other.deltat)*1e-6), \
            'sampling intervals differ %g, %g' % (self.deltat, other.delta)
        assert abs(self.tmin-other.tmin) < self.deltat*0.01, \
            'start times differ: %s, %s' % (
                util.time_to_str(self.tmin), util.time_to_str(other.tmin))
        assert abs(self.tmax-other.tmax) < self.deltat*0.01, \
            'end times differ: %s, %s' % (
                util.time_to_str(self.tmax), util.time_to_str(other.tmax))

        assert num.allclose(self.ydata, other.ydata, rtol=rtol, atol=atol), \
            'trace values differ'

    def __hash__(self):
        return id(self)

    def __call__(self, t, clip=False, snap=round):
        it = int(snap((t-self.tmin)/self.deltat))
        if clip:
            it = max(0, min(it, self.ydata.size-1))
        else:
            if it < 0 or self.ydata.size <= it:
                raise IndexError()

        return self.tmin+it*self.deltat, self.ydata[it]

    def interpolate(self, t, clip=False):
        '''
        Value of trace between supporting points through linear interpolation.

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
        '''
        Clip index to valid range.
        '''

        return min(max(0, i), self.ydata.size)

    def add(self, other, interpolate=True, left=0., right=0.):
        '''
        Add values of other trace (self += other).

        Add values of ``other`` trace to the values of ``self``, where it
        intersects with ``other``.  This method does not change the extent of
        ``self``. If ``interpolate`` is ``True`` (the default), the values of
        ``other`` to be added are interpolated at sampling instants of
        ``self``. Linear interpolation is performed. In this case the sampling
        rate of ``other`` must be equal to or lower than that of ``self``.  If
        ``interpolate`` is ``False``, the sampling rates of the two traces must
        match.
        '''

        if interpolate:
            assert self.deltat <= other.deltat \
                or same_sampling_rate(self, other)

            other_xdata = other.get_xdata()
            xdata = self.get_xdata()
            self.ydata += num.interp(
                xdata, other_xdata, other.ydata, left=left, right=left)
        else:
            assert self.deltat == other.deltat
            ioff = int(round((other.tmin-self.tmin)/self.deltat))
            ibeg = max(0, ioff)
            iend = min(self.data_len(), ioff+other.data_len())
            self.ydata[ibeg:iend] += other.ydata[ibeg-ioff:iend-ioff]

    def mult(self, other, interpolate=True):
        '''
        Muliply with values of other trace ``(self *= other)``.

        Multiply values of ``other`` trace to the values of ``self``, where it
        intersects with ``other``.  This method does not change the extent of
        ``self``. If ``interpolate`` is ``True`` (the default), the values of
        ``other`` to be multiplied are interpolated at sampling instants of
        ``self``. Linear interpolation is performed. In this case the sampling
        rate of ``other`` must be equal to or lower than that of ``self``.  If
        ``interpolate`` is ``False``, the sampling rates of the two traces must
        match.
        '''

        if interpolate:
            assert self.deltat <= other.deltat or \
                same_sampling_rate(self, other)

            other_xdata = other.get_xdata()
            xdata = self.get_xdata()
            self.ydata *= num.interp(
                xdata, other_xdata, other.ydata, left=0., right=0.)
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
        '''
        Get time and value of data maximum.
        '''

        i = num.argmax(self.ydata)
        return self.tmin + i*self.deltat, self.ydata[i]

    def min(self):
        '''
        Get time and value of data minimum.
        '''

        i = num.argmin(self.ydata)
        return self.tmin + i*self.deltat, self.ydata[i]

    def absmax(self):
        '''
        Get time and value of maximum of the absolute of data.
        '''

        tmi, mi = self.min()
        tma, ma = self.max()
        if abs(mi) > abs(ma):
            return tmi, abs(mi)
        else:
            return tma, abs(ma)

    def set_codes(
            self, network=None, station=None, location=None, channel=None,
            extra=None):

        '''
        Set network, station, location, and channel codes.
        '''

        if network is not None:
            self.network = network
        if station is not None:
            self.station = station
        if location is not None:
            self.location = location
        if channel is not None:
            self.channel = channel
        if extra is not None:
            self.extra = extra

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
        '''
        Check if trace has overlap with a given time span.
        '''
        return not (tmax < self.tmin or self.tmax < tmin)

    def is_relevant(self, tmin, tmax, selector=None):
        '''
        Check if trace has overlap with a given time span and matches a
        condition callback. (internal use)
        '''

        return not (tmax <= self.tmin or self.tmax < tmin) \
            and (selector is None or selector(self))

    def _update_ids(self):
        '''
        Update dependent ids.
        '''

        self.full_id = (
            self.network, self.station, self.location, self.channel, self.tmin)
        self.nslc_id = reuse(
            (self.network, self.station, self.location, self.channel))

    def prune_from_reuse_cache(self):
        util.deuse(self.nslc_id)
        util.deuse(self.network)
        util.deuse(self.station)
        util.deuse(self.location)
        util.deuse(self.channel)

    def set_mtime(self, mtime):
        '''
        Set modification time of the trace.
        '''

        self.mtime = mtime

    def get_xdata(self):
        '''
        Create array for time axis.
        '''

        if self.ydata is None:
            raise NoData()

        return self.tmin \
            + num.arange(len(self.ydata), dtype=num.float64) * self.deltat

    def get_ydata(self):
        '''
        Get data array.
        '''

        if self.ydata is None:
            raise NoData()

        return self.ydata

    def set_ydata(self, new_ydata):
        '''
        Replace data array.
        '''

        self.drop_growbuffer()
        self.ydata = new_ydata
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat

    def data_len(self):
        if self.ydata is not None:
            return self.ydata.size
        else:
            return int(round((self.tmax-self.tmin)/self.deltat)) + 1

    def drop_data(self):
        '''
        Forget data, make dataless trace.
        '''

        self.drop_growbuffer()
        self.ydata = None

    def drop_growbuffer(self):
        '''
        Detach the traces grow buffer.
        '''

        self._growbuffer = None
        self._pchain = None

    def copy(self, data=True):
        '''
        Make a deep copy of the trace.
        '''

        tracecopy = copy.copy(self)
        tracecopy.drop_growbuffer()
        if data:
            tracecopy.ydata = self.ydata.copy()
        tracecopy.meta = copy.deepcopy(self.meta)
        return tracecopy

    def crop_zeros(self):
        '''
        Remove any zeros at beginning and end.
        '''

        indices = num.where(self.ydata != 0.0)[0]
        if indices.size == 0:
            raise NoData()

        ibeg = indices[0]
        iend = indices[-1]+1
        if ibeg == 0 and iend == self.ydata.size-1:
            return

        self.drop_growbuffer()
        self.ydata = self.ydata[ibeg:iend].copy()
        self.tmin = self.tmin+ibeg*self.deltat
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        self._update_ids()

    def append(self, data):
        '''
        Append data to the end of the trace.

        To make this method efficient when successively very few or even single
        samples are appended, a larger grow buffer is allocated upon first
        invocation. The traces data is then changed to be a view into the
        currently filled portion of the grow buffer array.
        '''

        assert self.ydata.dtype == data.dtype
        newlen = data.size + self.ydata.size
        if self._growbuffer is None or self._growbuffer.size < newlen:
            self._growbuffer = num.empty(newlen*2, dtype=self.ydata.dtype)
            self._growbuffer[:self.ydata.size] = self.ydata
        self._growbuffer[self.ydata.size:newlen] = data
        self.ydata = self._growbuffer[:newlen]
        self.tmax = self.tmin + (newlen-1)*self.deltat

    def chop(
            self, tmin, tmax, inplace=True, include_last=False,
            snap=(round, round), want_incomplete=True):

        '''
        Cut the trace to given time span.

        If the ``inplace`` argument is True (the default) the trace is cut in
        place, otherwise a new trace with the cut part is returned.  By
        default, the indices where to start and end the trace data array are
        determined by rounding of ``tmin`` and ``tmax`` to sampling instances
        using Python's :py:func:`round` function. This behaviour can be changed
        with the ``snap`` argument, which takes a tuple of two functions (one
        for the lower and one for the upper end) to be used instead of
        :py:func:`round`.  The last sample is by default not included unless
        ``include_last`` is set to True.  If the given time span exceeds the
        available time span of the trace, the available part is returned,
        unless ``want_incomplete`` is set to False - in that case, a
        :py:exc:`NoData` exception is raised. This exception is always raised,
        when the requested time span does dot overlap with the trace's time
        span.
        '''

        if want_incomplete:
            if tmax <= self.tmin-self.deltat or self.tmax+self.deltat < tmin:
                raise NoData()
        else:
            if tmin < self.tmin or self.tmax < tmax:
                raise NoData()

        ibeg = max(0, t2ind(tmin-self.tmin, self.deltat, snap[0]))
        iplus = 0
        if include_last:
            iplus = 1

        iend = min(
            self.data_len(),
            t2ind(tmax-self.tmin, self.deltat, snap[1])+iplus)

        if ibeg >= iend:
            raise NoData()

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

    def downsample(
            self, ndecimate, snap=False, demean=False, ftype='fir-remez',
            cut=False):

        '''
        Downsample (decimate) trace by a given integer factor.

        Antialiasing filter details:

        * ``iir``: A Chebyshev type I digital filter of order 8 with maximum
          ripple of 0.05 dB in the passband.
        * ``fir``: A FIR filter using a Hamming window and 31 taps and a
          well-behaved phase response.
        * ``fir-remez``: A FIR filter based on the Remez exchange algorithm
          with ``45*ndecimate`` taps and a cutoff at 75% Nyquist frequency.

        Comparison of the digital filters:

        .. figure :: ../../static/downsampling-filter-comparison.png
            :width: 60%
            :alt: Comparison of the downsampling filters.

        See also :py:meth:`Trace.downsample_to`.

        :param ndecimate:
            Decimation factor, avoid values larger than 8.
        :type ndecimate:
            int

        :param snap:
            Whether to put the new sampling instants closest to multiples of
            the sampling rate (according to absolute time).
        :type snap:
            bool

        :param demean:
            Whether to demean the signal before filtering.
        :type demean:
            bool

        :param ftype:
            Which FIR filter to use, choose from ``'iir'``, ``'fir'``,
            ``'fir-remez'``. Default is ``'fir-remez'``.

        :param cut:
            Whether to cut off samples in the beginning of the trace which
            are polluted by artifacts of the anti-aliasing filter.
        :type cut:
            bool
        '''
        newdeltat = self.deltat*ndecimate
        b, a, n = util.decimate_coeffs(ndecimate, None, ftype)
        if snap:
            ilag = int(round((math.ceil(
                (self.tmin+(n//2 if cut else 0)*self.deltat) /
                newdeltat) * newdeltat - self.tmin) / self.deltat))
        else:
            ilag = (n//2 if cut else 0)

        data = self.ydata.astype(num.float64)
        if data.size != 0:
            if demean:
                data -= num.mean(data)
            y = signal.lfilter(b, a, data)
            self.ydata = y[n//2+ilag::ndecimate].copy()
        else:
            self.ydata = data

        self.tmin += ilag * self.deltat
        self.deltat = reuse(self.deltat*ndecimate)
        self.tmax = self.tmin+(len(self.ydata)-1)*self.deltat
        self._update_ids()

    def downsample_to(
            self, deltat, snap=False, allow_upsample_max=1, demean=False,
            ftype='fir-remez', cut=False):

        '''
        Downsample to given sampling rate.

        Tries to downsample the trace to a target sampling interval of
        ``deltat``. This runs :py:meth:`downsample` one or several times. If
        ``allow_upsample_max`` is set to a value larger than 1, intermediate
        upsampling steps are allowed, in order to increase the number of
        possible downsampling ratios.

        If the requested ratio is not supported, an exception of type
        :py:exc:`pyrocko.util.UnavailableDecimation` is raised.

        The downsampled trace will be shorter than the input trace because the
        anti-aliasing filter introduces edge effects. If `cut` is selected,
        additionally, polluted samples in the beginning of the trace are
        removed. The approximate amount of cutoff which will be produced by a
        given downsampling configuration can be estimated with
        :py:func:`downsample_tpad`.

        See also: :meth:`Trace.downsample`.

        :param deltat:
            Desired sampling interval in [s].
        :type deltat:
            float

        :param allow_upsample_max:
            Maximum allowed upsampling factor of the signal to achieve the
            desired sampling interval. Default is ``1``.
        :type allow_upsample_max:
            int

        :param snap:
            Whether to put the new sampling instants closest to multiples of
            the sampling rate (according to absolute time).
        :type snap:
            bool

        :param demean:
            Whether to demean the signal before filtering.
        :type demean:
            bool

        :param ftype:
            Which FIR filter to use, choose from ``'iir'``, ``'fir'``,
            ``'fir-remez'``. Default is ``'fir-remez'``.

        :param cut:
            Whether to cut off samples in the beginning of the trace which
            are polluted by artifacts of the anti-aliasing filter.
        :type demean:
            bool
        '''

        upsratio, deci_seq = _configure_downsampling(
            self.deltat, deltat, allow_upsample_max)

        if demean:
            self.drop_growbuffer()
            self.ydata = self.ydata.astype(num.float64)
            self.ydata -= num.mean(self.ydata)

        if upsratio > 1:
            self.drop_growbuffer()
            ydata = self.ydata
            self.ydata = num.zeros(
                ydata.size*upsratio-(upsratio-1), ydata.dtype)
            self.ydata[::upsratio] = ydata
            for i in range(1, upsratio):
                self.ydata[i::upsratio] = \
                    float(i)/upsratio * ydata[:-1] \
                    + float(upsratio-i)/upsratio * ydata[1:]
            self.deltat = self.deltat/upsratio

        for i, ndecimate in enumerate(deci_seq):
            self.downsample(
                ndecimate, snap=snap, demean=False, ftype=ftype, cut=cut)

    def resample(self, deltat):
        '''
        Resample to given sampling rate ``deltat``.

        Resampling is performed in the frequency domain.
        '''

        ndata = self.ydata.size
        ntrans = nextpow2(ndata)
        fntrans2 = ntrans * self.deltat/deltat
        ntrans2 = int(round(fntrans2))
        deltat2 = self.deltat * float(ntrans)/float(ntrans2)
        ndata2 = int(round(ndata*self.deltat/deltat2))
        if abs(fntrans2 - ntrans2) > 1e-7:
            logger.warning(
                'resample: requested deltat %g could not be matched exactly: '
                '%g' % (deltat, deltat2))

        data = self.ydata
        data_pad = num.zeros(ntrans, dtype=float)
        data_pad[:ndata] = data
        fdata = num.fft.rfft(data_pad)
        fdata2 = num.zeros((ntrans2+1)//2, dtype=fdata.dtype)
        n = min(fdata.size, fdata2.size)
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
            logger.warning(
                'resample_simple: less than one sample would have to be '
                'inserted/deleted per year. Doing nothing.')
            return

        ninterval = int(round(deltat / (self.deltat - deltat)))
        if abs(ninterval) < 20:
            logger.error(
                'resample_simple: sample insertion/deletion interval less '
                'than 20. results would be erroneous.')
            raise ResamplingFailed()

        delete = False
        if ninterval < 0:
            ninterval = - ninterval
            delete = True

        tyearbegin = util.year_start(self.tmin)

        nmin = int(round((self.tmin - tyearbegin)/deltat))

        ibegin = (((nmin-1)//ninterval)+1) * ninterval - nmin
        nindices = (len(self.ydata) - ibegin - 1) / ninterval + 1
        if nindices > 0:
            indices = ibegin + num.arange(nindices) * ninterval
            data_split = num.split(self.ydata, indices)
            data = []
            for ln, h in zip(data_split[:-1], data_split[1:]):
                if delete:
                    ln = ln[:-1]

                data.append(ln)
                if not delete:
                    if ln.size == 0:
                        v = h[0]
                    else:
                        v = 0.5*(ln[-1] + h[0])
                    data.append(num.array([v], dtype=ln.dtype))

            data.append(data_split[-1])

            ydata_new = num.concatenate(data)

            self.tmin = tyearbegin + nmin * deltat
            self.deltat = deltat
            self.set_ydata(ydata_new)

    def stretch(self, tmin_new, tmax_new):
        '''
        Stretch signal while preserving sample rate using sinc interpolation.

        :param tmin_new: new time of first sample
        :param tmax_new: new time of last sample

        This method can be used to correct for a small linear time drift or to
        introduce sub-sample time shifts. The amount of stretching is limited
        to 10% by the implementation and is expected to be much smaller than
        that by the approximations used.
        '''

        i_control = num.array([0, self.ydata.size-1], dtype=num.int64)
        t_control = num.array([tmin_new, tmax_new], dtype=float)

        r = (tmax_new - tmin_new) / self.deltat + 1.0
        n_new = int(round(r))
        if abs(n_new - r) > 0.001:
            n_new = int(math.floor(r))

        assert n_new >= 2

        tmax_new = tmin_new + (n_new-1) * self.deltat

        ydata_new = num.empty(n_new, dtype=float)
        signal_ext.antidrift(i_control, t_control,
                             self.ydata.astype(float),
                             tmin_new, self.deltat, ydata_new)

        self.tmin = tmin_new
        self.set_ydata(ydata_new)
        self._update_ids()

    def nyquist_check(self, frequency, intro='Corner frequency', warn=True,
                      raise_exception=False):

        '''
        Check if a given frequency is above the Nyquist frequency of the trace.

        :param intro: string used to introduce the warning/error message
        :param warn: whether to emit a warning
        :param raise_exception: whether to raise an :py:exc:`AboveNyquist`
            exception.
        '''

        if frequency >= 0.5/self.deltat:
            message = '%s (%g Hz) is equal to or higher than nyquist ' \
                      'frequency (%g Hz). (Trace %s)' \
                % (intro, frequency, 0.5/self.deltat, self.name())
            if warn:
                logger.warning(message)
            if raise_exception:
                raise AboveNyquist(message)

    def lowpass(self, order, corner, nyquist_warn=True,
                nyquist_exception=False, demean=True):

        '''
        Apply Butterworth lowpass to the trace.

        :param order: order of the filter
        :param corner: corner frequency of the filter

        Mean is removed before filtering.
        '''

        self.nyquist_check(
            corner, 'Corner frequency of lowpass', nyquist_warn,
            nyquist_exception)

        (b, a) = _get_cached_filter_coeffs(
            order, [corner*2.0*self.deltat], btype='low')

        if len(a) != order+1 or len(b) != order+1:
            logger.warning(
                'Erroneous filter coefficients returned by '
                'scipy.signal.butter(). You may need to downsample the '
                'signal before filtering.')

        data = self.ydata.astype(num.float64)
        if demean:
            data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b, a, data)

    def highpass(self, order, corner, nyquist_warn=True,
                 nyquist_exception=False, demean=True):

        '''
        Apply butterworth highpass to the trace.

        :param order: order of the filter
        :param corner: corner frequency of the filter

        Mean is removed before filtering.
        '''

        self.nyquist_check(
            corner, 'Corner frequency of highpass', nyquist_warn,
            nyquist_exception)

        (b, a) = _get_cached_filter_coeffs(
            order, [corner*2.0*self.deltat], btype='high')

        data = self.ydata.astype(num.float64)
        if len(a) != order+1 or len(b) != order+1:
            logger.warning(
                'Erroneous filter coefficients returned by '
                'scipy.signal.butter(). You may need to downsample the '
                'signal before filtering.')
        if demean:
            data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b, a, data)

    def bandpass(self, order, corner_hp, corner_lp, demean=True):
        '''
        Apply butterworth bandpass to the trace.

        :param order: order of the filter
        :param corner_hp: lower corner frequency of the filter
        :param corner_lp: upper corner frequency of the filter

        Mean is removed before filtering.
        '''

        self.nyquist_check(corner_hp, 'Lower corner frequency of bandpass')
        self.nyquist_check(corner_lp, 'Higher corner frequency of bandpass')
        (b, a) = _get_cached_filter_coeffs(
            order,
            [corner*2.0*self.deltat for corner in (corner_hp, corner_lp)],
            btype='band')
        data = self.ydata.astype(num.float64)
        if demean:
            data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b, a, data)

    def bandstop(self, order, corner_hp, corner_lp, demean=True):
        '''
        Apply bandstop (attenuates frequencies in band) to the trace.

        :param order: order of the filter
        :param corner_hp: lower corner frequency of the filter
        :param corner_lp: upper corner frequency of the filter

        Mean is removed before filtering.
        '''

        self.nyquist_check(corner_hp, 'Lower corner frequency of bandstop')
        self.nyquist_check(corner_lp, 'Higher corner frequency of bandstop')
        (b, a) = _get_cached_filter_coeffs(
            order,
            [corner*2.0*self.deltat for corner in (corner_hp, corner_lp)],
            btype='bandstop')
        data = self.ydata.astype(num.float64)
        if demean:
            data -= num.mean(data)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b, a, data)

    def envelope(self, inplace=True):
        '''
        Calculate the envelope of the trace.

        :param inplace: calculate envelope in place

        The calculation follows:

        .. math::

            Y' = \\sqrt{Y^2+H(Y)^2}

        where H is the Hilbert-Transform of the signal Y.
        '''

        y = self.ydata.astype(float)
        env = num.abs(hilbert(y))
        if inplace:
            self.drop_growbuffer()
            self.ydata = env
        else:
            tr = self.copy(data=False)
            tr.ydata = env
            return tr

    def taper(self, taperer, inplace=True, chop=False):
        '''
        Apply a :py:class:`Taper` to the trace.

        :param taperer: instance of :py:class:`Taper` subclass
        :param inplace: apply taper inplace
        :param chop: if ``True``: exclude tapered parts from the resulting
            trace
        '''

        if not inplace:
            tr = self.copy()
        else:
            tr = self

        if chop:
            i, n = taperer.span(tr.ydata, tr.tmin, tr.deltat)
            tr.shift(i*tr.deltat)
            tr.set_ydata(tr.ydata[i:i+n])

        taperer(tr.ydata, tr.tmin, tr.deltat)

        if not inplace:
            return tr

    def whiten(self, order=6):
        '''
        Whiten signal in time domain using autoregression and recursive filter.

        :param order: order of the autoregression process
        '''

        b, a = self.whitening_coefficients(order)
        self.drop_growbuffer()
        self.ydata = signal.lfilter(b, a, self.ydata)

    def whitening_coefficients(self, order=6):
        ar = yulewalker(self.ydata, order)
        b, a = [1.] + ar.tolist(), [1.]
        return b, a

    def ampspec_whiten(
            self,
            width,
            td_taper='auto',
            fd_taper='auto',
            pad_to_pow2=True,
            demean=True):

        '''
        Whiten signal via frequency domain using moving average on amplitude
        spectra.

        :param width: width of smoothing kernel [Hz]
        :param td_taper: time domain taper, object of type :py:class:`Taper` or
            ``None`` or ``'auto'``.
        :param fd_taper: frequency domain taper, object of type
            :py:class:`Taper` or ``None`` or ``'auto'``.
        :param pad_to_pow2: whether to pad the signal with zeros up to a length
            of 2^n
        :param demean: whether to demean the signal before tapering

        The signal is first demeaned and then tapered using ``td_taper``. Then,
        the spectrum is calculated and inversely weighted with a smoothed
        version of its amplitude spectrum. A moving average is used for the
        smoothing. The smoothed spectrum is then tapered using ``fd_taper``.
        Finally, the smoothed and tapered spectrum is back-transformed into the
        time domain.

        If ``td_taper`` is set to ``'auto'``, ``CosFader(1.0/width)`` is used.
        If ``fd_taper`` is set to ``'auto'``, ``CosFader(width)`` is used.
        '''

        ndata = self.data_len()

        if pad_to_pow2:
            ntrans = nextpow2(ndata)
        else:
            ntrans = ndata

        df = 1./(ntrans*self.deltat)
        nw = int(round(width/df))
        if ndata//2+1 <= nw:
            raise TraceTooShort(
                'Samples in trace: %s, samples needed: %s' % (ndata, nw))

        if td_taper == 'auto':
            td_taper = CosFader(1./width)

        if fd_taper == 'auto':
            fd_taper = CosFader(width)

        if td_taper:
            self.taper(td_taper)

        ydata = self.get_ydata().astype(float)
        if demean:
            ydata -= ydata.mean()

        spec = num.fft.rfft(ydata, ntrans)

        amp = num.abs(spec)
        nspec = amp.size
        csamp = num.cumsum(amp)
        amp_smoothed = num.empty(nspec, dtype=csamp.dtype)
        n1, n2 = nw//2, nw//2 + nspec - nw
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

        ydata = num.fft.irfft(spec)
        self.set_ydata(ydata[:ndata])

    def _get_cached_freqs(self, nf, deltaf):
        ck = (nf, deltaf)
        if ck not in Trace.cached_frequencies:
            Trace.cached_frequencies[ck] = deltaf * num.arange(
                nf, dtype=float)

        return Trace.cached_frequencies[ck]

    def bandpass_fft(self, corner_hp, corner_lp):
        '''
        Apply boxcar bandbpass to trace (in spectral domain).
        '''

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
        '''
        Time shift the trace.
        '''

        self.tmin += tshift
        self.tmax += tshift
        self._update_ids()

    def snap(self, inplace=True, interpolate=False):
        '''
        Shift trace samples to nearest even multiples of the sampling rate.

        :param inplace: (boolean) snap traces inplace

        If ``inplace`` is ``False`` and the difference of tmin and tmax of
        both, the snapped and the original trace is smaller than 0.01 x deltat,
        :py:func:`snap` returns the unsnapped instance of the original trace.
        '''

        tmin = round(self.tmin/self.deltat)*self.deltat
        tmax = tmin + (self.ydata.size-1)*self.deltat

        if inplace:
            xself = self
        else:
            if abs(self.tmin - tmin) < 1e-2*self.deltat and \
                    abs(self.tmax - tmax) < 1e-2*self.deltat:
                return self

            xself = self.copy()

        n = xself.data_len()
        if interpolate and n > 2:
            ydata_new = num.empty(n, dtype=float)
            i_control = num.array([0, n-1], dtype=num.int64)
            tref = tmin
            t_control = num.array(
                [float(xself.tmin-tref), float(xself.tmax-tref)],
                dtype=float)

            signal_ext.antidrift(i_control, t_control,
                                 xself.ydata.astype(float),
                                 float(tmin-tref), xself.deltat, ydata_new)

            xself.ydata = ydata_new

        xself.tmin = tmin
        xself.tmax = tmax
        xself._update_ids()

        return xself

    def fix_deltat_rounding_errors(self):
        '''
        Try to undo sampling rate rounding errors.

        See :py:func:`fix_deltat_rounding_errors`.
        '''

        self.deltat = fix_deltat_rounding_errors(self.deltat)
        self.tmax = self.tmin + (self.data_len() - 1) * self.deltat

    def sta_lta_centered(self, tshort, tlong, quad=True, scalingmethod=1):
        '''
        Run special STA/LTA filter where the short time window is centered on
        the long time window.

        :param tshort: length of short time window in [s]
        :param tlong: length of long time window in [s]
        :param quad: whether to square the data prior to applying the STA/LTA
            filter
        :param scalingmethod: integer key to select how output values are
            scaled / normalized (``1``, ``2``, or ``3``)

        ============= ====================================== ===========
        Scalingmethod Implementation                         Range
        ============= ====================================== ===========
        ``1``         As/Al* Ts/Tl                           [0,1]
        ``2``         (As/Al - 1) / (Tl/Ts - 1)              [-Ts/Tl,1]
        ``3``         Like ``2`` but clipping range at zero  [0,1]
        ============= ====================================== ===========

        '''

        nshort = int(round(tshort/self.deltat))
        nlong = int(round(tlong/self.deltat))

        assert nshort < nlong
        if nlong > len(self.ydata):
            raise TraceTooShort(
                'Samples in trace: %s, samples needed: %s'
                % (len(self.ydata), nlong))

        if quad:
            sqrdata = self.ydata**2
        else:
            sqrdata = self.ydata

        mavg_short = moving_avg(sqrdata, nshort)
        mavg_long = moving_avg(sqrdata, nlong)

        self.drop_growbuffer()

        if scalingmethod not in (1, 2, 3):
            raise Exception('Invalid argument to scalingrange argument.')

        if scalingmethod == 1:
            self.ydata = mavg_short/mavg_long * float(nshort)/float(nlong)
        elif scalingmethod in (2, 3):
            self.ydata = (mavg_short/mavg_long - 1.) \
                / ((float(nlong)/float(nshort)) - 1)

        if scalingmethod == 3:
            self.ydata = num.maximum(self.ydata, 0.)

    def sta_lta_right(self, tshort, tlong, quad=True, scalingmethod=1):
        '''
        Run special STA/LTA filter where the short time window is overlapping
        with the last part of the long time window.

        :param tshort: length of short time window in [s]
        :param tlong: length of long time window in [s]
        :param quad: whether to square the data prior to applying the STA/LTA
            filter
        :param scalingmethod: integer key to select how output values are
            scaled / normalized (``1``, ``2``, or ``3``)

        ============= ====================================== ===========
        Scalingmethod Implementation                         Range
        ============= ====================================== ===========
        ``1``         As/Al* Ts/Tl                           [0,1]
        ``2``         (As/Al - 1) / (Tl/Ts - 1)              [-Ts/Tl,1]
        ``3``         Like ``2`` but clipping range at zero  [0,1]
        ============= ====================================== ===========

        With ``scalingmethod=1``, the values produced by this variant of the
        STA/LTA are equivalent to

        .. math::
            s_i = \\frac{s}{l} \\frac{\\frac{1}{s}\\sum_{j=i}^{i+s-1} f_j}
                                     {\\frac{1}{l}\\sum_{j=i+s-l}^{i+s-1} f_j}

        where :math:`f_j` are the input samples, :math:`s` are the number of
        samples in the short time window and :math:`l` are the number of
        samples in the long time window.
        '''

        n = self.data_len()
        tmin = self.tmin

        nshort = max(1, int(round(tshort/self.deltat)))
        nlong = max(1, int(round(tlong/self.deltat)))

        assert nshort < nlong

        if nlong > len(self.ydata):
            raise TraceTooShort(
                'Samples in trace: %s, samples needed: %s'
                % (len(self.ydata), nlong))

        if quad:
            sqrdata = self.ydata**2
        else:
            sqrdata = self.ydata

        nshift = int(math.floor(0.5 * (nlong - nshort)))
        if nlong % 2 != 0 and nshort % 2 == 0:
            nshift += 1

        mavg_short = moving_avg(sqrdata, nshort)[nshift:]
        mavg_long = moving_avg(sqrdata, nlong)[:sqrdata.size-nshift]

        self.drop_growbuffer()

        if scalingmethod not in (1, 2, 3):
            raise Exception('Invalid argument to scalingrange argument.')

        if scalingmethod == 1:
            ydata = mavg_short/mavg_long * nshort/nlong
        elif scalingmethod in (2, 3):
            ydata = (mavg_short/mavg_long - 1.) \
                / ((float(nlong)/float(nshort)) - 1)

        if scalingmethod == 3:
            ydata = num.maximum(ydata, 0.)

        self.set_ydata(ydata)

        self.shift((math.ceil(0.5*nlong) - nshort + 1) * self.deltat)

        self.chop(
            tmin + (nlong - nshort) * self.deltat,
            tmin + (n - nshort) * self.deltat)

    def peaks(self, threshold, tsearch,
              deadtime=False,
              nblock_duration_detection=100):

        '''
        Detect peaks above a given threshold (method 1).

        From every instant, where the signal rises above ``threshold``, a time
        length of ``tsearch`` seconds is searched for a maximum. A list with
        tuples (time, value) for each detected peak is returned. The
        ``deadtime`` argument turns on a special deadtime duration detection
        algorithm useful in combination with recursive STA/LTA filters.
        '''

        y = self.ydata
        above = num.where(y > threshold, 1, 0)
        deriv = num.zeros(y.size, dtype=num.int8)
        deriv[1:] = above[1:]-above[:-1]
        itrig_positions = num.nonzero(deriv > 0)[0]
        tpeaks = []
        apeaks = []
        tzeros = []
        tzero = self.tmin

        for itrig_pos in itrig_positions:
            ibeg = itrig_pos
            iend = min(
                len(self.ydata),
                itrig_pos + int(math.ceil(tsearch/self.deltat)))
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
                        tzero = self.tmin + (len(y)-1) * self.deltat
                        break

                    logy = num.log(
                        y[ibeg+iblock*nblock:ibeg+(iblock+1)*nblock])
                    logy[0] += totalsum
                    ysum = num.cumsum(logy)
                    totalsum = ysum[-1]
                    below = num.where(ysum <= 0., 1, 0)
                    deriv = num.zeros(ysum.size, dtype=num.int8)
                    deriv[1:] = below[1:]-below[:-1]
                    izero_positions = num.nonzero(deriv > 0)[0] + iblock*nblock
                    if len(izero_positions) > 0:
                        tzero = self.tmin + self.deltat * (
                            ibeg + izero_positions[0])
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

    def peaks2(self, threshold, tsearch):

        '''
        Detect peaks above a given threshold (method 2).

        This variant of peak detection is a bit more robust (and slower) than
        the one implemented in :py:meth:`Trace.peaks`. First all samples with
        ``a[i-1] < a[i] > a[i+1]`` are masked as potential peaks. From these,
        iteratively the one with the maximum amplitude ``a[j]`` and time
        ``t[j]`` is choosen and potential peaks within
        ``t[j] - tsearch, t[j] + tsearch``
        are discarded. The algorithm stops, when ``a[j] < threshold`` or when
        no more potential peaks are left.
        '''

        a = num.copy(self.ydata)

        amin = num.min(a)

        a[0] = amin
        a[1: -1][num.diff(a, 2) <= 0.] = amin
        a[-1] = amin

        data = []
        while True:
            imax = num.argmax(a)
            amax = a[imax]

            if amax < threshold or amax == amin:
                break

            data.append((self.tmin + imax * self.deltat, amax))

            ntsearch = int(round(tsearch / self.deltat))
            a[max(imax-ntsearch//2, 0):min(imax+ntsearch//2, a.size)] = amin

        if data:
            data.sort()
            tpeaks, apeaks = list(zip(*data))
        else:
            tpeaks, apeaks = [], []

        return tpeaks, apeaks

    def extend(self, tmin=None, tmax=None, fillmethod='zeros'):
        '''
        Extend trace to given span.

        :param tmin: begin time of new span
        :param tmax: end time of new span
        :param fillmethod: ``'zeros'``,  ``'repeat'``, ``'mean'``, or
            ``'median'``
        '''

        nold = self.ydata.size

        if tmin is not None:
            nl = min(0, int(round((tmin-self.tmin)/self.deltat)))
        else:
            nl = 0

        if tmax is not None:
            nh = max(nold - 1, int(round((tmax-self.tmin)/self.deltat)))
        else:
            nh = nold - 1

        n = nh - nl + 1
        data = num.zeros(n, dtype=self.ydata.dtype)
        data[-nl:-nl + nold] = self.ydata
        if self.ydata.size >= 1:
            if fillmethod == 'repeat':
                data[:-nl] = self.ydata[0]
                data[-nl + nold:] = self.ydata[-1]
            elif fillmethod == 'median':
                v = num.median(self.ydata)
                data[:-nl] = v
                data[-nl + nold:] = v
            elif fillmethod == 'mean':
                v = num.mean(self.ydata)
                data[:-nl] = v
                data[-nl + nold:] = v

        self.drop_growbuffer()
        self.ydata = data

        self.tmin += nl * self.deltat
        self.tmax = self.tmin + (self.ydata.size - 1) * self.deltat

        self._update_ids()

    def transfer(self,
                 tfade=0.,
                 freqlimits=None,
                 transfer_function=None,
                 cut_off_fading=True,
                 demean=True,
                 invert=False):

        '''
        Return new trace with transfer function applied (convolution).

        :param tfade: rise/fall time in seconds of taper applied in timedomain
            at both ends of trace.
        :param freqlimits: 4-tuple with corner frequencies in Hz.
        :param transfer_function: FrequencyResponse object; must provide a
            method 'evaluate(freqs)', which returns the transfer function
            coefficients at the frequencies 'freqs'.
        :param cut_off_fading: whether to cut off rise/fall interval in output
            trace.
        :param demean: remove mean before applying transfer function
        :param invert: set to True to do a deconvolution
        '''

        if transfer_function is None:
            transfer_function = g_one_response

        if self.tmax - self.tmin <= tfade*2.:
            raise TraceTooShort(
                'Trace %s.%s.%s.%s too short for fading length setting. '
                'trace length = %g, fading length = %g'
                % (self.nslc_id + (self.tmax-self.tmin, tfade)))

        if freqlimits is None and (
                transfer_function is None or transfer_function.is_scalar()):

            # special case for flat responses

            output = self.copy()
            data = self.ydata
            ndata = data.size

            if transfer_function is not None:
                c = num.abs(transfer_function.evaluate(num.ones(1))[0])

                if invert:
                    c = 1.0/c

                data *= c

            if tfade != 0.0:
                data *= costaper(
                    0., tfade, self.deltat*(ndata-1)-tfade, self.deltat*ndata,
                    ndata, self.deltat)

            output.ydata = data

        else:
            ndata = self.ydata.size
            ntrans = nextpow2(ndata*1.2)
            coeffs = self._get_tapered_coeffs(
                ntrans, freqlimits, transfer_function, invert=invert,
                demean=demean)

            data = self.ydata

            data_pad = num.zeros(ntrans, dtype=float)
            data_pad[:ndata] = data
            if demean:
                data_pad[:ndata] -= data.mean()

            if tfade != 0.0:
                data_pad[:ndata] *= costaper(
                    0., tfade, self.deltat*(ndata-1)-tfade, self.deltat*ndata,
                    ndata, self.deltat)

            fdata = num.fft.rfft(data_pad)
            fdata *= coeffs
            ddata = num.fft.irfft(fdata)
            output = self.copy()
            output.ydata = ddata[:ndata]

        if cut_off_fading and tfade != 0.0:
            try:
                output.chop(output.tmin+tfade, output.tmax-tfade, inplace=True)
            except NoData:
                raise TraceTooShort(
                    'Trace %s.%s.%s.%s too short for fading length setting. '
                    'trace length = %g, fading length = %g'
                    % (self.nslc_id + (self.tmax-self.tmin, tfade)))
        else:
            output.ydata = output.ydata.copy()

        return output

    def differentiate(self, n=1, order=4, inplace=True):
        '''
        Approximate first or second derivative of the trace.

        :param n: 1 for first derivative, 2 for second
        :param order: order of the approximation 2 and 4 are supported
        :param inplace: if ``True`` the trace is differentiated in place,
            otherwise a new trace object with the derivative is returned.

        Raises :py:exc:`ValueError` for unsupported `n` or `order`.

        See :py:func:`~pyrocko.util.diff_fd` for implementation details.
        '''

        ddata = util.diff_fd(n, order, self.deltat, self.ydata)

        if inplace:
            self.ydata = ddata
        else:
            output = self.copy(data=False)
            output.set_ydata(ddata)
            return output

    def drop_chain_cache(self):
        if self._pchain:
            self._pchain.clear()

    def init_chain(self):
        self._pchain = pchain.Chain(
            do_downsample,
            do_extend,
            do_pre_taper,
            do_fft,
            do_filter,
            do_ifft)

    def run_chain(self, tmin, tmax, deltat, setup, nocache):
        if setup.domain == 'frequency_domain':
            _, _, data = self._pchain(
                (self, deltat),
                (tmin, tmax),
                (setup.taper,),
                (setup.filter,),
                (setup.filter,),
                nocache=nocache)

            return num.abs(data), num.abs(data)

        else:
            processed = self._pchain(
                (self, deltat),
                (tmin, tmax),
                (setup.taper,),
                (setup.filter,),
                (setup.filter,),
                (),
                nocache=nocache)

            if setup.domain == 'time_domain':
                data = processed.get_ydata()

            elif setup.domain == 'envelope':
                processed = processed.envelope(inplace=False)

            elif setup.domain == 'absolute':
                processed.set_ydata(num.abs(processed.get_ydata()))

            return processed.get_ydata(), processed

    def misfit(self, candidate, setup, nocache=False, debug=False):
        '''
        Calculate misfit and normalization factor against candidate trace.

        :param candidate: :py:class:`Trace` object
        :param setup: :py:class:`MisfitSetup` object
        :returns: tuple ``(m, n)``, where m is the misfit value and n is the
            normalization divisor

        If the sampling rates of ``self`` and ``candidate`` differ, the trace
        with the higher sampling rate will be downsampled.
        '''

        a = self
        b = candidate

        for tr in (a, b):
            if not tr._pchain:
                tr.init_chain()

        deltat = max(a.deltat, b.deltat)
        tmin = min(a.tmin, b.tmin) - deltat
        tmax = max(a.tmax, b.tmax) + deltat

        adata, aproc = a.run_chain(tmin, tmax, deltat, setup, nocache)
        bdata, bproc = b.run_chain(tmin, tmax, deltat, setup, nocache)

        if setup.domain != 'cc_max_norm':
            m, n = Lx_norm(bdata, adata, norm=setup.norm)
        else:
            ctr = correlate(aproc, bproc, mode='full', normalization='normal')
            ccmax = ctr.max()[1]
            m = 0.5 - 0.5 * ccmax
            n = 0.5

        if debug:
            return m, n, aproc, bproc
        else:
            return m, n

    def spectrum(self, pad_to_pow2=False, tfade=None, ntrans_min=None):
        '''
        Get FFT spectrum of trace.

        :param pad_to_pow2: whether to zero-pad the data to next larger
            power-of-two length
        :param tfade: ``None`` or a time length in seconds, to apply cosine
            shaped tapers to both

        :returns: a tuple with (frequencies, values)
        '''

        if ntrans_min is None:
            ndata = self.ydata.size
        else:
            ndata = ntrans_min

        if pad_to_pow2:
            ntrans = nextpow2(ndata)
        else:
            ntrans = ndata

        if tfade is None:
            ydata = self.ydata
        else:
            ydata = self.ydata * costaper(
                0., tfade, self.deltat*(ndata-1)-tfade, self.deltat*ndata,
                ndata, self.deltat)

        fydata = num.fft.rfft(ydata, ntrans)
        df = 1./(ntrans*self.deltat)
        fxdata = num.arange(len(fydata))*df
        return fxdata, fydata

    def multi_filter(self, filter_freqs, bandwidth):

        class Gauss(FrequencyResponse):
            f0 = Float.T()
            a = Float.T(default=1.0)

            def __init__(self, f0, a=1.0, **kwargs):
                FrequencyResponse.__init__(self, f0=f0, a=a, **kwargs)

            def evaluate(self, freqs):
                omega0 = 2.*math.pi*self.f0
                omega = 2.*math.pi*freqs
                return num.exp(-((omega-omega0)
                                 / (self.a*omega0))**2)

        freqs, coeffs = self.spectrum()
        n = self.data_len()
        nfilt = len(filter_freqs)
        signal_tf = num.zeros((nfilt, n))
        centroid_freqs = num.zeros(nfilt)
        for ifilt, f0 in enumerate(filter_freqs):
            taper = Gauss(f0, a=bandwidth)
            weights = taper.evaluate(freqs)
            nhalf = freqs.size
            analytic_spec = num.zeros(n, dtype=complex)
            analytic_spec[:nhalf] = coeffs*weights

            enorm = num.abs(analytic_spec[:nhalf])**2
            enorm /= num.sum(enorm)

            if n % 2 == 0:
                analytic_spec[1:nhalf-1] *= 2.
            else:
                analytic_spec[1:nhalf] *= 2.

            analytic = num.fft.ifft(analytic_spec)
            signal_tf[ifilt, :] = num.abs(analytic)

            enorm = num.abs(analytic_spec[:nhalf])**2
            enorm /= num.sum(enorm)
            centroid_freqs[ifilt] = num.sum(freqs*enorm)

        return centroid_freqs, signal_tf

    def _get_tapered_coeffs(
            self, ntrans, freqlimits, transfer_function, invert=False,
            demean=True):

        cache_key = (
            ntrans, self.deltat, freqlimits, transfer_function.uuid, invert,
            demean)

        if cache_key in g_tapered_coeffs_cache:
            return g_tapered_coeffs_cache[cache_key]

        deltaf = 1./(self.deltat*ntrans)
        nfreqs = ntrans//2 + 1
        transfer = num.ones(nfreqs, dtype=complex)
        hi = snapper(nfreqs, deltaf)
        if freqlimits is not None:
            kmin, kmax = hi(freqlimits[0]), hi(freqlimits[3])
            freqs = num.arange(kmin, kmax)*deltaf
            coeffs = transfer_function.evaluate(freqs)
            if invert:
                if num.any(coeffs == 0.0):
                    raise InfiniteResponse('%s.%s.%s.%s' % self.nslc_id)

                transfer[kmin:kmax] = 1.0 / coeffs
            else:
                transfer[kmin:kmax] = coeffs

            tapered_transfer = costaper(*freqlimits, nfreqs, deltaf) * transfer
        else:
            if invert:
                raise Exception(
                    'transfer: `freqlimits` must be given when `invert` is '
                    'set to `True`')

            freqs = num.arange(nfreqs) * deltaf
            tapered_transfer = transfer_function.evaluate(freqs)

        g_tapered_coeffs_cache[cache_key] = tapered_transfer

        if demean:
            tapered_transfer[0] = 0.0  # don't introduce static offsets

        return tapered_transfer

    def fill_template(self, template, **additional):
        '''
        Fill string template with trace metadata.

        Uses normal python '%(placeholder)s' string templates. The following
        placeholders are considered: ``network``, ``station``, ``location``,
        ``channel``, ``tmin`` (time of first sample), ``tmax`` (time of last
        sample), ``tmin_ms``, ``tmax_ms``, ``tmin_us``, ``tmax_us``,
        ``tmin_year``, ``tmax_year``, ``tmin_month``, ``tmax_month``,
        ``tmin_day``, ``tmax_day``, ``julianday``. The variants ending with
        ``'_ms'`` include milliseconds, those with ``'_us'`` include
        microseconds, those with ``'_year'`` contain only the year.
        '''

        template = template.replace('%n', '%(network)s')\
            .replace('%s', '%(station)s')\
            .replace('%l', '%(location)s')\
            .replace('%c', '%(channel)s')\
            .replace('%b', '%(tmin)s')\
            .replace('%e', '%(tmax)s')\
            .replace('%j', '%(julianday)s')

        return template % TraceStringFiller(self, additional=additional)

    def plot(self):
        '''
        Show trace with matplotlib.

        See also: :py:meth:`Trace.snuffle`.
        '''

        import pylab
        pylab.plot(self.get_xdata(), self.get_ydata())
        name = '%s %s %s - %s' % (
            self.channel,
            self.station,
            time.strftime('%d-%m-%y %H:%M:%S', time.gmtime(self.tmin)),
            time.strftime('%d-%m-%y %H:%M:%S', time.gmtime(self.tmax)))

        pylab.title(name)
        pylab.show()

    def snuffle(self, **kwargs):
        '''
        Show trace in a snuffler window.

        :param stations: list of :py:class:`pyrocko.model.station.Station`
            objects or ``None``
        :param events: list of :py:class:`pyrocko.model.event.Event` objects or
            ``None``
        :param markers: list of :py:class:`pyrocko.gui.snuffler.marker.Marker`
            objects or ``None``
        :param ntracks: float, number of tracks to be shown initially (default:
            12)
        :param follow: time interval (in seconds) for real time follow mode or
            ``None``
        :param controls: bool, whether to show the main controls (default:
            ``True``)
        :param opengl: bool, whether to use opengl (default: ``False``)
        '''

        return snuffle([self], **kwargs)


def snuffle(traces, **kwargs):
    '''
    Show traces in a snuffler window.

    :param stations: list of :py:class:`pyrocko.model.station.Station` objects
        or ``None``
    :param events: list of :py:class:`pyrocko.model.event.Event` objects or
        ``None``
    :param markers: list of :py:class:`pyrocko.gui.snuffler.marker.Marker`
        objects or ``None``
    :param ntracks: int, number of tracks to be shown initially (default: 12)
    :param follow: time interval (in seconds) for real time follow mode or
        ``None``
    :param controls: bool, whether to show the main controls (default:
        ``True``)
    :param opengl: bool, whether to use opengl (default: ``False``)
    '''

    from pyrocko import pile
    from pyrocko.gui.snuffler import snuffler
    p = pile.Pile()
    if traces:
        trf = pile.MemTracesFile(None, traces)
        p.add_file(trf)
    return snuffler.snuffle(p, **kwargs)


def downsample_tpad(
        deltat_in, deltat_out, allow_upsample_max=1, ftype='fir-remez'):
    '''
    Get approximate amount of cutoff which will be produced by downsampling.

    The  :py:meth:`Trace.downsample_to` method removes some samples at the
    beginning and end of the trace which is downsampled. This function
    estimates the approximate length [s] which will be cut off for a given set
    of parameters supplied to :py:meth:`Trace.downsample_to`.

    :param deltat_in:
        Input sampling interval [s].
    :type deltat_in:
        float

    :param deltat_out:
        Output samling interval [s].
    :type deltat_out:
        float

    :returns:
        Approximate length [s] which will be cut off.

    See :py:meth:`Trace.downsample_to` for details.
    '''

    upsratio, deci_seq = _configure_downsampling(
        deltat_in, deltat_out, allow_upsample_max)

    tpad = 0.0
    deltat = deltat_in / upsratio
    for deci in deci_seq:
        b, a, n = util.decimate_coeffs(deci, None, ftype)
        # n//2 for the antialiasing
        # +deci for possible snap to multiples
        # +1 for rounding errors
        tpad += (n//2 + deci + 1) * deltat
        deltat = deltat * deci

    return tpad


def _configure_downsampling(deltat_in, deltat_out, allow_upsample_max):
    for upsratio in range(1, allow_upsample_max+1):
        dratio = (upsratio/deltat_in) / (1./deltat_out)
        deci_seq = util.decitab(int(round(dratio)))
        if abs(dratio - round(dratio)) / dratio < 0.0001 and deci_seq:
            return upsratio, [deci for deci in deci_seq if deci != 1]

    raise util.UnavailableDecimation('ratio = %g' % (deltat_out / deltat_in))


def _all_same(xs):
    return all(x == xs[0] for x in xs)


def _incompatibilities(traces):
    if not traces:
        return None

    params = [
        (tr.ydata.size, tr.ydata.dtype, tr.deltat, tr.tmin)
        for tr in traces]

    if not _all_same(params):
        return params
    else:
        return None


def _raise_incompatible_traces(params):
    raise IncompatibleTraces(
        'Given traces are incompatible. Sampling rate, start time, '
        'number of samples and data type must match.\n%s\n%s' % (
            '  %10s %-10s %12s %22s' % (
                'nsamples', 'dtype', 'deltat', 'tmin'),
            '\n'.join(
                '  %10i %-10s %12.5e %22s' % (
                    nsamples, dtype, deltat, util.time_to_str(tmin))
                for (nsamples, dtype, deltat, tmin) in params)))


def _ensure_compatible(traces):
    params = _incompatibilities(traces)
    if params:
        _raise_incompatible_traces(params)


def _almost_equal(a, b, atol):
    return abs(a-b) < atol


def get_traces_data_as_array(traces):
    '''
    Merge data samples from multiple traces into a 2D array.

    :param traces:
        Input waveforms.
    :type traces:
        list of :py:class:`pyrocko.Trace <pyrocko.trace.Trace>` objects

    :raises:
        :py:class:`IncompatibleTraces` if traces have different time
        span, sample rate or data type, or if traces is an empty list.

    :returns:
        2D array as ``data[itrace, isample]``.
    :rtype:
        :py:class:`numpy.ndarray`
    '''

    if not traces:
        raise ValueError('Need at least one trace.')

    _ensure_compatible(traces)

    return num.vstack([tr.ydata for tr in traces])


def _ensure_aligned(traces):
    if not traces:
        raise ValueError('No traces given.')

    eps = 1e-3
    deltats = sorted(set(tr.deltat for tr in traces))
    if len(deltats) != 1:
        raise UnalignedTraces(
            'Differing sampling intervals: %s' % ', '.join(
                str(deltat) for deltat in deltats))

    dtypes = sorted(set(tr.ydata.dtype for tr in traces))
    if len(dtypes) != 1:
        raise UnalignedTraces(
            'Differing data types: %s' % ', '.join(
                str(dtype) for dtype in dtypes))

    deltat = deltats[0]
    tmins = num.array([tr.tmin for tr in traces])
    toffsets = num.abs(num.round(tmins / deltat) * deltat - tmins)
    is_aligned = toffsets < deltat * eps
    if not all(is_aligned):
        info = '\n  start times:\n' + '\n'.join(
            '    %s: %s %g' % (tr.codes, util.time_to_str(tr.tmin), tr.deltat)
            for tr in traces)

        raise UnalignedTraces(
            'Samples of some traces are not aligned: %s' % (
                ', '.join(str(tr.codes) for tr in [
                    traces[i] for i in num.where(
                        num.logical_not(is_aligned))[0]])) + info)

    return None


def merge_traces_data_as_array(traces, tmin=None, tmax=None, codes=None):
    from numpy.ma import masked_array
    from pyrocko.squirrel.model import WAVEFORM, codes_patterns_for_kind

    if codes is not None:
        codes = codes_patterns_for_kind(WAVEFORM, codes)

    if not traces:
        raise NoData('Need at least one trace.')

    _ensure_aligned(traces)

    if codes is None:
        codes = sorted(set(tr.codes for tr in traces))
    else:
        _codes = set(codes)
        assert len(_codes) == len(codes)
        traces = [tr for tr in traces if tr.codes in _codes]
        if not traces:
            raise ValueError('Need at least one trace.')

    codes_to_i = dict((codes, i) for (i, codes) in enumerate(codes))

    if tmax is None:
        tmax = max(tr.tmax + tr.deltat for tr in traces)

    if tmin is None:
        tmin = min(tr.tmin for tr in traces)

    deltat = traces[0].deltat

    nsamples = int(round((tmax - tmin) / deltat))

    data = num.zeros(
        (len(codes), nsamples),
        dtype=traces[0].ydata.dtype)

    mask = num.ones(data.shape, dtype=bool)
    for tr in traces:
        itmax = nsamples
        itmin_tr = int(round((tr.tmin - tmin) / deltat))
        itmax_tr = itmin_tr + tr.ydata.size
        itmin_common = max(0, itmin_tr)
        itmax_common = min(itmax, itmax_tr)
        icodes = codes_to_i[tr.codes]
        data[icodes, itmin_common:itmax_common] \
            = tr.ydata[itmin_common-itmin_tr:itmax_common-itmin_tr]
        mask[icodes, itmin_common:itmax_common] = False

    return masked_array(data, mask=mask), codes, tmin, deltat


def make_traces_compatible(
        traces,
        dtype=None,
        deltat=None,
        enforce_global_snap=True,
        warn_snap=False,
        chop=True):

    '''
    Homogenize sampling rate, time span, sampling instants, and data type.

    This function takes a group of traces and tries to make them compatible in
    terms of data type and sampling rate, time span, and sampling instants of
    time.

    If necessary, traces are (in order):

    - casted to the same data type.
    - downsampled to a common sampling rate, using decimation cascades.
    - resampled to common sampling instants in time, using Sinc interpolation.
    - cut to the same time span. The longest time span covered by all traces is
      used.

    :param traces:
        Input waveforms.
    :type traces:
        :py:class:`list` of :py:class:`Trace`

    :param dtype:
        Force traces to be casted to the given data type. If not specified, the
        traces are cast to :py:class:`float`.
    :type dtype:
        :py:class:`numpy.dtype`

    :param deltat:
        Sampling interval [s]. If not specified, the longest sampling interval
        among the input traces is chosen.
    :type deltat:
        float

    :param enforce_global_snap:
        If ``True``, choose sampling instants to be even multiples of the
        sampling rate in system time. When set to ``False`` traces are still
        resampled to common time instants (if necessary), but they may be
        offset to the system time sampling rate multiples.
    :type enforce_global_snap:
        bool

    :param warn_snap:
        If set to ``True`` warn, when resampling has to be performed.
    :type warn_snap:
        bool
    '''

    eps_snap = 1e-3

    if not traces:
        return []

    traces = list(traces)

    dtypes = [tr.ydata.dtype for tr in traces]
    if not _all_same(dtypes) or dtype is not None:

        if dtype is None:
            dtype = float
            logger.warning(
                'make_traces_compatible: Inconsistent data types - converting '
                'sample datatype to %s.' % str(dtype))

        for itr, tr in enumerate(traces):
            tr_copy = tr.copy(data=False)
            tr_copy.set_ydata(tr.ydata.astype(dtype))
            traces[itr] = tr_copy

    if dtype is None:
        dtype = dtypes[0]

    deltats = [tr.deltat for tr in traces]
    if not _all_same(deltats) or deltat is not None:
        if deltat is None:
            deltat = max(deltats)
            logger.warning(
                'make_traces_compatible: Inconsistent sampling rates - '
                'downsampling to lowest rate among input traces: %g Hz.'
                % (1.0 / deltat))

        for itr, tr in enumerate(traces):
            if tr.deltat != deltat:
                tr_copy = tr.copy()
                tr_copy.downsample_to(deltat, snap=True, cut=True)
                if tr_copy.ydata.dtype is not dtype:
                    tr_copy.set_ydata(tr_copy.ydata.astype(dtype))

                traces[itr] = tr_copy

    if deltat is None:
        deltat = deltats[0]

    tmins = num.array([tr.tmin for tr in traces])
    is_aligned = num.abs(num.round(tmins / deltat) * deltat - tmins) \
        > deltat * eps_snap

    if enforce_global_snap or any(is_aligned):
        tref = util.to_time_float(0.0)
    else:
        # to keep a common subsample shift
        tref = num.max(tmins)

    tmins_snap = num.round((tmins - tref) / deltat) * deltat + tref
    need_snap = num.abs(tmins_snap - tmins) > deltat * eps_snap
    if num.any(need_snap):
        if warn_snap:
            logger.warning(
                'make_traces_compatible: Misaligned sampling - introducing '
                'subsample shifts for proper alignment.')

        for itr, tr in enumerate(traces):
            if need_snap[itr]:
                tr_copy = tr.copy()
                if tref != 0.0:
                    tr_copy.shift(-tref)

                tr_copy.snap(interpolate=True)
                if tr_copy.ydata.dtype is not dtype:
                    tr_copy.set_ydata(tr_copy.ydata.astype(dtype))

                if tref != 0.0:
                    tr_copy.shift(tref)

                traces[itr] = tr_copy

    tmins = num.array([tr.tmin for tr in traces])
    nsamples = num.array([tr.ydata.size for tr in traces])
    tmaxs = tmins + (nsamples - 1) * deltat

    tmin = num.max(tmins)
    tmax = num.min(tmaxs)

    if not chop:
        return traces

    if tmin > tmax:
        raise IncompatibleTraces('Traces do not overlap.')

    nsamples_must = int(round((tmax - tmin) / deltat)) + 1
    for itr, tr in enumerate(traces):
        if not (_almost_equal(tr.tmin, tmin, deltat*eps_snap)
                and _almost_equal(tr.tmax, tmax, deltat*eps_snap)):

            traces[itr] = tr.chop(
                tmin, tmax,
                inplace=False,
                want_incomplete=False,
                include_last=True)

        xtr = traces[itr]
        assert _almost_equal(xtr.tmin, tmin, deltat*eps_snap)
        assert int(round((xtr.tmax - xtr.tmin) / deltat)) + 1 == nsamples_must
        xtr.tmin = tmin
        xtr.tmax = tmax
        xtr.deltat = deltat
        xtr._update_ids()

    return traces


class IncompatibleTraces(Exception):
    '''
    Raised when traces have incompatible sampling rate, time span or data type.
    '''


class UnalignedTraces(Exception):
    '''
    Raised when traces have incompatible sampling rate, time span or data type.
    '''


class InfiniteResponse(Exception):
    '''
    This exception is raised by :py:class:`Trace` operations when deconvolution
    of a frequency response (instrument response transfer function) would
    result in a division by zero.
    '''


class MisalignedTraces(Exception):
    '''
    This exception is raised by some :py:class:`Trace` operations when tmin,
    tmax or number of samples do not match.
    '''

    pass


class OverlappingTraces(Exception):
    '''
    This exception is raised by some :py:class:`Trace` operations when traces
    overlap but should not.
    '''

    pass


class NoData(Exception):
    '''
    This exception is raised by some :py:class:`Trace` operations when no or
    not enough data is available.
    '''

    pass


class AboveNyquist(Exception):
    '''
    This exception is raised by some :py:class:`Trace` operations when given
    frequencies are above the Nyquist frequency.
    '''

    pass


class TraceTooShort(Exception):
    '''
    This exception is raised by some :py:class:`Trace` operations when the
    trace is too short.
    '''

    pass


class ResamplingFailed(Exception):
    pass


class TraceStringFiller:

    def __init__(self, tr, additional={}):
        self.tr = tr
        self.additional = additional

    def __getitem__(self, k):
        if k in ('network', 'station', 'location', 'channel', 'extra'):
            return getattr(self.tr, k)

        method = getattr(self, 'get_' + k, None)
        if method:
            return method()

        return self.additional[k]

    def _filename_safe(self, s):
        return re.sub(r'[^0-9A-Za-z_-]', '_', s)

    def get_network_safe(self):
        return self._filename_safe(self.tr.network)

    def get_station_safe(self):
        return self._filename_safe(self.tr.station)

    def get_location_safe(self):
        return self._filename_safe(self.tr.location)

    def get_channel_safe(self):
        return self._filename_safe(self.tr.channel)

    def get_extra_safe(self):
        return self._filename_safe(self.tr.extra)

    def get_network_dsafe(self):
        return self._filename_safe(self.tr.network) or '_'

    def get_station_dsafe(self):
        return self._filename_safe(self.tr.station) or '_'

    def get_location_dsafe(self):
        return self._filename_safe(self.tr.location) or '_'

    def get_channel_dsafe(self):
        return self._filename_safe(self.tr.channel) or '_'

    def get_extra_dsafe(self):
        return self._filename_safe(self.tr.extra) or '_'

    def get_tmin(self):
        return util.time_to_str(self.tr.tmin, format='%Y-%m-%d_%H-%M-%S')

    def get_tmax(self):
        return util.time_to_str(self.tr.tmax, format='%Y-%m-%d_%H-%M-%S')

    def get_tmin_ms(self):
        return util.time_to_str(self.tr.tmin, format='%Y-%m-%d_%H-%M-%S.3FRAC')

    def get_tmax_ms(self):
        return util.time_to_str(self.tr.tmax, format='%Y-%m-%d_%H-%M-%S.3FRAC')

    def get_tmin_us(self):
        return util.time_to_str(self.tr.tmin, format='%Y-%m-%d_%H-%M-%S.6FRAC')

    def get_tmax_us(self):
        return util.time_to_str(self.tr.tmax, format='%Y-%m-%d_%H-%M-%S.6FRAC')

    def get_tmin_year(self):
        return util.time_to_str(self.tr.tmin, format='%Y')

    def get_tmin_month(self):
        return util.time_to_str(self.tr.tmin, format='%m')

    def get_tmin_day(self):
        return util.time_to_str(self.tr.tmin, format='%d')

    def get_tmax_year(self):
        return util.time_to_str(self.tr.tmax, format='%Y')

    def get_tmax_month(self):
        return util.time_to_str(self.tr.tmax, format='%m')

    def get_tmax_day(self):
        return util.time_to_str(self.tr.tmax, format='%d')

    def get_julianday(self):
        return str(util.julian_day_of_year(self.tr.tmin))

    def get_tmin_jday(self):
        return util.time_to_str(self.tr.tmin, format='%j')

    def get_tmax_jday(self):
        return util.time_to_str(self.tr.tmax, format='%j')


def minmax(traces, key=None, mode='minmax', outer_mode='minmax'):

    '''
    Get data range given traces grouped by selected pattern.

    :param key: a callable which takes as single argument a trace and returns a
        key for the grouping of the results. If this is ``None``, the default,
        ``lambda tr: (tr.network, tr.station, tr.location, tr.channel)`` is
        used.
    :param mode: ``'minmax'`` or floating point number. If this is
        ``'minmax'``, minimum and maximum of the traces are used, if it is a
        number, mean +- standard deviation times ``mode`` is used.

    param outer_mode: ``'minmax'`` to use mininum and maximum of the
        single-trace ranges, or ``'robust'`` to use the interval to discard 10%
        extreme values on either end.

    :returns: a dict with the combined data ranges.

    Examples::

        ranges = minmax(traces, lambda tr: tr.channel)
        print ranges['N']   # print min & max of all traces with channel == 'N'
        print ranges['E']   # print min & max of all traces with channel == 'E'

        ranges = minmax(traces, lambda tr: (tr.network, tr.station))
        print ranges['GR', 'HAM3']  # print min & max of all traces with
                                    # network == 'GR' and station == 'HAM3'

        ranges = minmax(traces, lambda tr: None)
        print ranges[None]  # prints min & max of all traces
    '''

    if key is None:
        key = _default_key

    ranges = defaultdict(list)
    for trace in traces:
        if isinstance(mode, str) and mode == 'minmax':
            mi, ma = num.nanmin(trace.ydata), num.nanmax(trace.ydata)
        else:
            mean = trace.ydata.mean()
            std = trace.ydata.std()
            mi, ma = mean-std*mode, mean+std*mode

        k = key(trace)
        ranges[k].append((mi, ma))

    for k in ranges:
        mins, maxs = num.array(ranges[k]).T
        if outer_mode == 'minmax':
            ranges[k] = num.nanmin(mins), num.nanmax(maxs)
        elif outer_mode == 'robust':
            ranges[k] = num.percentile(mins, 10.), num.percentile(maxs, 90.)

    return ranges


def minmaxtime(traces, key=None):

    '''
    Get time range given traces grouped by selected pattern.

    :param key: a callable which takes as single argument a trace and returns a
        key for the grouping of the results. If this is ``None``, the default,
        ``lambda tr: (tr.network, tr.station, tr.location, tr.channel)`` is
        used.

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
            ranges[k] = min(tmi, mi), max(tma, ma)

    return ranges


def degapper(
        traces,
        maxgap=5,
        fillmethod='interpolate',
        deoverlap='use_second',
        maxlap=None):

    '''
    Try to connect traces and remove gaps.

    This method will combine adjacent traces, which match in their network,
    station, location and channel attributes. Overlapping parts are handled
    according to the ``deoverlap`` argument.

    :param traces: input traces, must be sorted by their full_id attribute.
    :param maxgap: maximum number of samples to interpolate.
    :param fillmethod: what to put into the gaps: 'interpolate' or 'zeros'.
    :param deoverlap: how to handle overlaps: 'use_second' to use data from
        second trace (default), 'use_first' to use data from first trace,
        'crossfade_cos' to crossfade with cosine taper, 'add' to add amplitude
        values.
    :param maxlap:      maximum number of samples of overlap which are removed

    :returns:           list of traces
    '''

    in_traces = traces
    out_traces = []
    if not in_traces:
        return out_traces
    out_traces.append(in_traces.pop(0))
    while in_traces:

        a = out_traces[-1]
        b = in_traces.pop(0)

        avirt, bvirt = a.ydata is None, b.ydata is None
        assert avirt == bvirt, \
            'traces given to degapper() must either all have data or have ' \
            'no data.'

        virtual = avirt and bvirt

        if (a.nslc_id == b.nslc_id and a.deltat == b.deltat
                and a.data_len() >= 1 and b.data_len() >= 1
                and (virtual or a.ydata.dtype == b.ydata.dtype)):

            dist = (b.tmin-(a.tmin+(a.data_len()-1)*a.deltat))/a.deltat
            idist = int(round(dist))
            if abs(dist - idist) > 0.05 and idist <= maxgap:
                # logger.warning('Cannot degap traces with displaced sampling '
                #                '(%s, %s, %s, %s)' % a.nslc_id)

                if idist <= 0 and (maxlap is None or -maxlap < idist):
                    # still cut off overlap (cut off on first trace)
                    try:
                        a.chop(a.tmin, max(a.tmin, b.tmin-b.deltat))
                    except NoData:
                        pass

                pass

            else:
                if 1 < idist <= maxgap:
                    if not virtual:
                        if fillmethod == 'interpolate':
                            filler = a.ydata[-1] + (
                                ((1.0 + num.arange(idist-1, dtype=float))
                                 / idist) * (b.ydata[0]-a.ydata[-1])
                            ).astype(a.ydata.dtype)
                        elif fillmethod == 'zeros':
                            filler = num.zeros(idist-1, dtype=a.ydata.dtype)
                        a.ydata = num.concatenate((a.ydata, filler, b.ydata))
                    a.tmax = b.tmax
                    if a.mtime and b.mtime:
                        a.mtime = max(a.mtime, b.mtime)
                    continue

                elif idist == 1:
                    if not virtual:
                        a.ydata = num.concatenate((a.ydata, b.ydata))
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
                                a.ydata = num.concatenate(
                                    (a.ydata[:-n], b.ydata))
                            elif deoverlap in ('use_first', 'crossfade_cos'):
                                a.ydata = num.concatenate(
                                    (a.ydata, b.ydata[n:]))
                            elif deoverlap == 'add':
                                a.ydata[-n:] += b.ydata[:n]
                                a.ydata = num.concatenate(
                                    (a.ydata, b.ydata[n:]))
                            else:
                                assert False, 'unknown deoverlap method'

                            if deoverlap == 'crossfade_cos':
                                n = -idist+1
                                taper = 0.5-0.5*num.cos(
                                    (1.+num.arange(n))/(1.+n)*num.pi)
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
    '''
    2D rotation of traces.

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
            if ((a.channel, b.channel) == in_channels and
                    a.nslc_id[:3] == b.nslc_id[:3] and
                    abs(a.deltat-b.deltat) < a.deltat*0.001):
                tmin = max(a.tmin, b.tmin)
                tmax = min(a.tmax, b.tmax)

                if tmin < tmax:
                    ac = a.chop(tmin, tmax, inplace=False, include_last=True)
                    bc = b.chop(tmin, tmax, inplace=False, include_last=True)
                    if abs(ac.tmin - bc.tmin) > ac.deltat*0.01:
                        logger.warning(
                            'Cannot rotate traces with displaced sampling '
                            '(%s, %s, %s, %s)' % a.nslc_id)
                        continue

                    acydata = ac.get_ydata()*cphi+bc.get_ydata()*sphi
                    bcydata = -ac.get_ydata()*sphi+bc.get_ydata()*cphi
                    ac.set_ydata(acydata)
                    bc.set_ydata(bcydata)

                    ac.set_codes(channel=out_channels[0])
                    bc.set_codes(channel=out_channels[1])
                    rotated.append(ac)
                    rotated.append(bc)

    return rotated


def rotate_to_rt(n, e, source, receiver, out_channels=('R', 'T')):
    '''
    Rotate traces from NE to RT system.

    :param n:
        North trace.
    :type n:
        :py:class:`~pyrocko.trace.Trace`

    :param e:
        East trace.
    :type e:
        :py:class:`~pyrocko.trace.Trace`

    :param source:
        Source of the recorded signal.
    :type source:
        :py:class:`pyrocko.gf.seismosizer.Source`

    :param receiver:
        Receiver of the recorded signal.
    :type receiver:
        :py:class:`pyrocko.model.location.Location`

    :param out_channels:
        Channel codes of the output channels (radial, transversal).
        Default is ('R', 'T').

    :type out_channels
        optional, tuple[str, str]

    :returns:
        Rotated traces (radial, transversal).
    :rtype:
        tuple[
            :py:class:`~pyrocko.trace.Trace`,
            :py:class:`~pyrocko.trace.Trace`]
    '''
    azimuth = orthodrome.azimuth(receiver, source) + 180.
    in_channels = n.channel, e.channel
    out = rotate(
        [n, e], azimuth,
        in_channels=in_channels,
        out_channels=out_channels)

    assert len(out) == 2
    for tr in out:
        if tr.channel == out_channels[0]:
            r = tr
        elif tr.channel == out_channels[1]:
            t = tr
        else:
            assert False

    return r, t


def rotate_to_lqt(traces, backazimuth, incidence, in_channels,
                  out_channels=('L', 'Q', 'T')):
    '''
    Rotate traces from ZNE to LQT system.

    :param traces: list of traces in arbitrary order
    :param backazimuth: backazimuth in degrees clockwise from north
    :param incidence: incidence angle in degrees from vertical
    :param in_channels: input channel names
    :param out_channels: output channel names (default: ('L', 'Q', 'T'))
    :returns: list of transformed traces
    '''
    i = incidence/180.*num.pi
    b = backazimuth/180.*num.pi

    ci = num.cos(i)
    cb = num.cos(b)
    si = num.sin(i)
    sb = num.sin(b)

    rotmat = num.array(
        [[ci, -cb*si, -sb*si], [si, cb*ci, sb*ci], [0., sb, -cb]])
    return project(traces, rotmat, in_channels, out_channels)


def _decompose(a):
    '''
    Decompose matrix into independent submatrices.
    '''

    def depends(iout, a):
        row = a[iout, :]
        return set(num.arange(row.size).compress(row != 0.0))

    def provides(iin, a):
        col = a[:, iin]
        return set(num.arange(col.size).compress(col != 0.0))

    a = num.asarray(a)
    outs = set(range(a.shape[0]))
    systems = []
    while outs:
        iout = outs.pop()

        gout = set()
        for iin in depends(iout, a):
            gout.update(provides(iin, a))

        if not gout:
            continue

        gin = set()
        for iout2 in gout:
            gin.update(depends(iout2, a))

        if not gin:
            continue

        for iout2 in gout:
            if iout2 in outs:
                outs.remove(iout2)

        gin = list(gin)
        gin.sort()
        gout = list(gout)
        gout.sort()

        systems.append((gin, gout, a[gout, :][:, gin]))

    return systems


def _channels_to_names(channels):
    from pyrocko import squirrel
    names = []
    for ch in channels:
        if isinstance(ch, model.Channel):
            names.append(ch.name)
        elif isinstance(ch, squirrel.Channel):
            names.append(ch.codes.channel)
        else:
            names.append(ch)

    return names


def project(traces, matrix, in_channels, out_channels):
    '''
    Affine transform of three-component traces.

    Compute matrix-vector product of three-component traces, to e.g. rotate
    traces into a different basis. The traces are distinguished and ordered by
    their channel attribute. The tranform is applied to overlapping parts of
    any appropriate combinations of the input traces. This should allow this
    function to be robust with data gaps. It also tries to apply the
    tranformation to subsets of the channels, if this is possible, so that, if
    for example a vertical compontent is missing, horizontal components can
    still be rotated.

    :param traces: list of traces in arbitrary order
    :param matrix: tranformation matrix
    :param in_channels: input channel names
    :param out_channels: output channel names
    :returns: list of transformed traces
    '''

    in_channels = tuple(_channels_to_names(in_channels))
    out_channels = tuple(_channels_to_names(out_channels))
    systems = _decompose(matrix)

    # fallback to full matrix if some are not quadratic
    for iins, iouts, submatrix in systems:
        if submatrix.shape[0] != submatrix.shape[1]:
            if len(in_channels) != 3 or len(out_channels) != 3:
                return []
            else:
                return _project3(traces, matrix, in_channels, out_channels)

    projected = []
    for iins, iouts, submatrix in systems:
        in_cha = tuple([in_channels[iin] for iin in iins])
        out_cha = tuple([out_channels[iout] for iout in iouts])
        if submatrix.shape[0] == 1:
            projected.extend(_project1(traces, submatrix, in_cha, out_cha))
        elif submatrix.shape[1] == 2:
            projected.extend(_project2(traces, submatrix, in_cha, out_cha))
        else:
            projected.extend(_project3(traces, submatrix, in_cha, out_cha))

    return projected


def project_dependencies(matrix, in_channels, out_channels):
    '''
    Figure out what dependencies project() would produce.
    '''

    in_channels = tuple(_channels_to_names(in_channels))
    out_channels = tuple(_channels_to_names(out_channels))
    systems = _decompose(matrix)

    subpro = []
    for iins, iouts, submatrix in systems:
        if submatrix.shape[0] != submatrix.shape[1]:
            subpro.append((matrix, in_channels, out_channels))

    if not subpro:
        for iins, iouts, submatrix in systems:
            in_cha = tuple([in_channels[iin] for iin in iins])
            out_cha = tuple([out_channels[iout] for iout in iouts])
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
    assert matrix.shape == (1, 1)

    projected = []
    for a in traces:
        if not (a.channel,) == in_channels:
            continue

        ac = a.copy()
        ac.set_ydata(matrix[0, 0]*a.get_ydata())
        ac.set_codes(channel=out_channels[0])
        projected.append(ac)

    return projected


def _project2(traces, matrix, in_channels, out_channels):
    assert len(in_channels) == 2
    assert len(out_channels) == 2
    assert matrix.shape == (2, 2)
    projected = []
    for a in traces:
        for b in traces:
            if not ((a.channel, b.channel) == in_channels and
                    a.nslc_id[:3] == b.nslc_id[:3] and
                    abs(a.deltat-b.deltat) < a.deltat*0.001):
                continue

            tmin = max(a.tmin, b.tmin)
            tmax = min(a.tmax, b.tmax)

            if tmin > tmax:
                continue

            ac = a.chop(tmin, tmax, inplace=False, include_last=True)
            bc = b.chop(tmin, tmax, inplace=False, include_last=True)
            if abs(ac.tmin - bc.tmin) > ac.deltat*0.01:
                logger.warning(
                    'Cannot project traces with displaced sampling '
                    '(%s, %s, %s, %s)' % a.nslc_id)
                continue

            acydata = num.dot(matrix[0], (ac.get_ydata(), bc.get_ydata()))
            bcydata = num.dot(matrix[1], (ac.get_ydata(), bc.get_ydata()))

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
    assert matrix.shape == (3, 3)
    projected = []
    for a in traces:
        for b in traces:
            for c in traces:
                if not ((a.channel, b.channel, c.channel) == in_channels
                        and a.nslc_id[:3] == b.nslc_id[:3]
                        and b.nslc_id[:3] == c.nslc_id[:3]
                        and abs(a.deltat-b.deltat) < a.deltat*0.001
                        and abs(b.deltat-c.deltat) < b.deltat*0.001):

                    continue

                tmin = max(a.tmin, b.tmin, c.tmin)
                tmax = min(a.tmax, b.tmax, c.tmax)

                if tmin >= tmax:
                    continue

                ac = a.chop(tmin, tmax, inplace=False, include_last=True)
                bc = b.chop(tmin, tmax, inplace=False, include_last=True)
                cc = c.chop(tmin, tmax, inplace=False, include_last=True)
                if (abs(ac.tmin - bc.tmin) > ac.deltat*0.01
                        or abs(bc.tmin - cc.tmin) > bc.deltat*0.01):

                    logger.warning(
                        'Cannot project traces with displaced sampling '
                        '(%s, %s, %s, %s)' % a.nslc_id)
                    continue

                acydata = num.dot(
                    matrix[0],
                    (ac.get_ydata(), bc.get_ydata(), cc.get_ydata()))
                bcydata = num.dot(
                    matrix[1],
                    (ac.get_ydata(), bc.get_ydata(), cc.get_ydata()))
                ccydata = num.dot(
                    matrix[2],
                    (ac.get_ydata(), bc.get_ydata(), cc.get_ydata()))

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
    '''
    Cross correlation of two traces.

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

    assert_same_sampling_rate(a, b)

    ya, yb = a.ydata, b.ydata

    # need reversed order here:
    yc = numpy_correlate_fixed(yb, ya, mode=mode, use_fft=use_fft)
    kmin, kmax = numpy_correlate_lag_range(yb, ya, mode=mode, use_fft=use_fft)

    if normalization == 'normal':
        normfac = num.sqrt(num.sum(ya**2))*num.sqrt(num.sum(yb**2))
        yc = yc/normfac

    elif normalization == 'gliding':
        if mode != 'valid':
            assert False, 'gliding normalization currently only available ' \
                'with "valid" mode.'

        if ya.size < yb.size:
            yshort, ylong = ya, yb
        else:
            yshort, ylong = yb, ya

        epsilon = 0.00001
        normfac_short = num.sqrt(num.sum(yshort**2))
        normfac = normfac_short * num.sqrt(
            moving_sum(ylong**2, yshort.size, mode='valid')) \
            + normfac_short*epsilon

        if yb.size <= ya.size:
            normfac = normfac[::-1]

        yc /= normfac

    c = a.copy()
    c.set_ydata(yc)
    c.set_codes(*merge_codes(a, b, '~'))
    c.shift(-c.tmin + b.tmin-a.tmin + kmin * c.deltat)

    return c


def deconvolve(
        a, b, waterlevel,
        tshift=0.,
        pad=0.5,
        fd_taper=None,
        pad_to_pow2=True):

    same_sampling_rate(a, b)
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
    denom = num.maximum(bautocorr, waterlevel * bautocorr.max())

    out /= denom
    df = 1/(ntrans*deltat)

    if fd_taper is not None:
        fd_taper(out, 0.0, df)

    ydata = num.roll(num.fft.irfft(out), int(round(tshift/deltat)))
    c = a.copy(data=False)
    c.set_ydata(ydata[:ndata])
    c.set_codes(*merge_codes(a, b, '/'))
    return c


def assert_same_sampling_rate(a, b, eps=1.0e-6):
    assert same_sampling_rate(a, b, eps), \
        'Sampling rates differ: %g != %g' % (a.deltat, b.deltat)


def same_sampling_rate(a, b, eps=1.0e-6):
    '''
    Check if two traces have the same sampling rate.

    :param a,b: input traces
    :param eps: relative tolerance
    '''

    return abs(a.deltat - b.deltat) < (a.deltat + b.deltat)*eps


def fix_deltat_rounding_errors(deltat):
    '''
    Try to undo sampling rate rounding errors.

    Fix rounding errors of sampling intervals when these are read from single
    precision floating point values.

    Assumes that the true sampling rate or sampling interval was an integer
    value. No correction will be applied if this would change the sampling
    rate by more than 0.001%.
    '''

    if deltat <= 1.0:
        deltat_new = 1.0 / round(1.0 / deltat)
    else:
        deltat_new = round(deltat)

    if abs(deltat_new - deltat) / deltat > 1e-5:
        deltat_new = deltat

    return deltat_new


def merge_codes(a, b, sep='-'):
    '''
    Merge network-station-location-channel codes of a pair of traces.
    '''

    o = []
    for xa, xb in zip(a.nslc_id, b.nslc_id):
        if xa == xb:
            o.append(xa)
        else:
            o.append(sep.join((xa, xb)))
    return o


class Taper(Object):
    '''
    Base class for tapers.

    Does nothing by default.
    '''

    def __call__(self, y, x0, dx):
        pass


class CosTaper(Taper):
    '''
    Cosine Taper.

    :param a: start of fading in
    :param b: end of fading in
    :param c: start of fading out
    :param d: end of fading out
    '''

    a = Float.T()
    b = Float.T()
    c = Float.T()
    d = Float.T()

    def __init__(self, a, b, c, d):
        Taper.__init__(self, a=a, b=b, c=c, d=d)

    def __call__(self, y, x0, dx):

        if y.dtype == num.dtype(float):
            _apply_costaper = signal_ext.apply_costaper
        else:
            _apply_costaper = apply_costaper

        _apply_costaper(self.a, self.b, self.c, self.d, y, x0, dx)

    def span(self, y, x0, dx):
        return span_costaper(self.a, self.b, self.c, self.d, y, x0, dx)

    def time_span(self):
        return self.a, self.d


class CosFader(Taper):
    '''
    Cosine Fader.

    :param xfade: fade in and fade out time in seconds (optional)
    :param xfrac: fade in and fade out as fraction between 0. and 1. (optional)

    Only one argument can be set. The other should to be ``None``.
    '''

    xfade = Float.T(optional=True)
    xfrac = Float.T(optional=True)

    def __init__(self, xfade=None, xfrac=None):
        Taper.__init__(self, xfade=xfade, xfrac=xfrac)
        assert (xfade is None) != (xfrac is None)
        self._xfade = xfade
        self._xfrac = xfrac

    def __call__(self, y, x0, dx):

        xfade = self._xfade

        xlen = (y.size - 1)*dx
        if xfade is None:
            xfade = xlen * self._xfrac

        a = x0
        b = x0 + xfade
        c = x0 + xlen - xfade
        d = x0 + xlen

        apply_costaper(a, b, c, d, y, x0, dx)

    def span(self, y, x0, dx):
        return 0, y.size

    def time_span(self):
        return None, None


def none_min(li):
    if None in li:
        return None
    else:
        return min(x for x in li if x is not None)


def none_max(li):
    if None in li:
        return None
    else:
        return max(x for x in li if x is not None)


class MultiplyTaper(Taper):
    '''
    Multiplication of several tapers.
    '''

    tapers = List.T(Taper.T())

    def __init__(self, tapers=None):
        if tapers is None:
            tapers = []

        Taper.__init__(self, tapers=tapers)

    def __call__(self, y, x0, dx):
        for taper in self.tapers:
            taper(y, x0, dx)

    def span(self, y, x0, dx):
        spans = []
        for taper in self.tapers:
            spans.append(taper.span(y, x0, dx))

        mins, maxs = list(zip(*spans))
        return min(mins), max(maxs)

    def time_span(self):
        spans = []
        for taper in self.tapers:
            spans.append(taper.time_span())

        mins, maxs = list(zip(*spans))
        return none_min(mins), none_max(maxs)


class GaussTaper(Taper):
    '''
    Frequency domain Gaussian filter.
    '''

    alpha = Float.T()

    def __init__(self, alpha):
        Taper.__init__(self, alpha=alpha)
        self._alpha = alpha

    def __call__(self, y, x0, dx):
        f = x0 + num.arange(y.size)*dx
        y *= num.exp(-num.pi**2 / (self._alpha**2) * f**2)


cached_coefficients = {}


def _get_cached_filter_coeffs(order, corners, btype):
    ck = (order, tuple(corners), btype)
    if ck not in cached_coefficients:
        if len(corners) == 1:
            corners = corners[0]

        cached_coefficients[ck] = signal.butter(
            order, corners, btype=btype)

    return cached_coefficients[ck]


class _globals(object):
    _numpy_has_correlate_flip_bug = None


def _default_key(tr):
    return (tr.network, tr.station, tr.location, tr.channel)


def numpy_has_correlate_flip_bug():
    '''
    Check if NumPy's correlate function reveals old behaviour.
    '''

    if _globals._numpy_has_correlate_flip_bug is None:
        a = num.array([0, 0, 1, 0, 0, 0, 0])
        b = num.array([0, 0, 0, 0, 1, 0, 0, 0])
        ab = num.correlate(a, b, mode='same')
        ba = num.correlate(b, a, mode='same')
        _globals._numpy_has_correlate_flip_bug = num.all(ab == ba)

    return _globals._numpy_has_correlate_flip_bug


def numpy_correlate_fixed(a, b, mode='valid', use_fft=False):
    '''
    Call :py:func:`numpy.correlate` with fixes.

        c[k] = sum_i a[i+k] * conj(b[i])

    Note that the result produced by newer numpy.correlate is always flipped
    with respect to the formula given in its documentation (if ascending k
    assumed for the output).
    '''

    if use_fft:
        if a.size < b.size:
            c = signal.fftconvolve(b[::-1], a, mode=mode)
        else:
            c = signal.fftconvolve(a, b[::-1], mode=mode)
        return c

    else:
        buggy = numpy_has_correlate_flip_bug()

        a = num.asarray(a)
        b = num.asarray(b)

        if buggy:
            b = num.conj(b)

        c = num.correlate(a, b, mode=mode)

        if buggy and a.size < b.size:
            return c[::-1]
        else:
            return c


def numpy_correlate_emulate(a, b, mode='valid'):
    '''
    Slow version of :py:func:`numpy.correlate` for comparison.
    '''

    a = num.asarray(a)
    b = num.asarray(b)
    kmin = -(b.size-1)
    klen = a.size-kmin
    kmin, kmax = numpy_correlate_lag_range(a, b, mode=mode)
    kmin = int(kmin)
    kmax = int(kmax)
    klen = kmax - kmin + 1
    c = num.zeros(klen, dtype=num.promote_types(b.dtype, a.dtype))
    for k in range(kmin, kmin+klen):
        imin = max(0, -k)
        ilen = min(b.size, a.size-k) - imin
        c[k-kmin] = num.sum(
            a[imin+k:imin+ilen+k] * num.conj(b[imin:imin+ilen]))

    return c


def numpy_correlate_lag_range(a, b, mode='valid', use_fft=False):
    '''
    Get range of lags for which :py:func:`numpy.correlate` produces values.
    '''

    a = num.asarray(a)
    b = num.asarray(b)

    kmin = -(b.size-1)
    if mode == 'full':
        klen = a.size-kmin
    elif mode == 'same':
        klen = max(a.size, b.size)
        kmin += (a.size+b.size-1 - max(a.size, b.size)) // 2 + \
            int(not use_fft and a.size % 2 == 0 and b.size > a.size)
    elif mode == 'valid':
        klen = abs(a.size - b.size) + 1
        kmin += min(a.size, b.size) - 1

    return kmin, kmin + klen - 1


def autocorr(x, nshifts):
    '''
    Compute biased estimate of the first autocorrelation coefficients.

    :param x: input array
    :param nshifts: number of coefficients to calculate
    '''

    mean = num.mean(x)
    std = num.std(x)
    n = x.size
    xdm = x - mean
    r = num.zeros(nshifts)
    for k in range(nshifts):
        r[k] = 1./((n-num.abs(k))*std) * num.sum(xdm[:n-k] * xdm[k:])

    return r


def yulewalker(x, order):
    '''
    Compute autoregression coefficients using Yule-Walker method.

    :param x: input array
    :param order: number of coefficients to produce

    A biased estimate of the autocorrelation is used. The Yule-Walker equations
    are solved by :py:func:`numpy.linalg.inv` instead of Levinson-Durbin
    recursion which is normally used.
    '''

    gamma = autocorr(x, order+1)
    d = gamma[1:1+order]
    a = num.zeros((order, order))
    gamma2 = num.concatenate((gamma[::-1], gamma[1:order]))
    for i in range(order):
        ioff = order-i
        a[i, :] = gamma2[ioff:ioff+order]

    return num.dot(num.linalg.inv(a), -d)


def moving_avg(x, n):
    n = int(n)
    cx = x.cumsum()
    nn = len(x)
    y = num.zeros(nn, dtype=cx.dtype)
    y[n//2:n//2+(nn-n)] = (cx[n:]-cx[:-n])/n
    y[:n//2] = y[n//2]
    y[n//2+(nn-n):] = y[n//2+(nn-n)-1]
    return y


def moving_sum(x, n, mode='valid'):
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
        n1 = (n-1)//2
        y = num.zeros(nn, dtype=cx.dtype)
        if n <= nn:
            y[0:n-n1] = cx[n1:n]
            y[n-n1:nn-n1] = cx[n:nn]-cx[0:nn-n]
            y[nn-n1:nn] = cx[nn-1] - cx[nn-n:nn-n+n1]
        else:
            y[0:max(0, nn-n1)] = cx[min(n1, nn):nn]
            y[max(nn-n1, 0):min(n-n1, nn)] = cx[nn-1]
            y[min(n-n1, nn):nn] = cx[nn-1] - cx[0:max(0, nn-(n-n1))]

    return y


def nextpow2(i):
    return 2**int(math.ceil(math.log(i)/math.log(2.)))


def snapper_w_offset(nmax, offset, delta, snapfun=math.ceil):
    def snap(x):
        return max(0, min(int(snapfun((x-offset)/delta)), nmax))
    return snap


def snapper(nmax, delta, snapfun=math.ceil):
    def snap(x):
        return max(0, min(int(snapfun(x/delta)), nmax))
    return snap


def apply_costaper(a, b, c, d, y, x0, dx):
    abcd = num.array((a, b, c, d), dtype=float)
    ja, jb, jc, jd = num.clip(num.ceil((abcd-x0)/dx).astype(int), 0, y.size)
    y[:ja] = 0.
    y[ja:jb] *= 0.5 \
        - 0.5*num.cos((dx*num.arange(ja, jb)-(a-x0))/(b-a)*num.pi)
    y[jc:jd] *= 0.5 \
        + 0.5*num.cos((dx*num.arange(jc, jd)-(c-x0))/(d-c)*num.pi)
    y[jd:] = 0.


def span_costaper(a, b, c, d, y, x0, dx):
    hi = snapper_w_offset(y.size, x0, dx)
    return hi(a), hi(d) - hi(a)


def costaper(a, b, c, d, nfreqs, deltaf):
    hi = snapper(nfreqs, deltaf)
    tap = num.zeros(nfreqs)
    tap[hi(a):hi(b)] = 0.5 \
        - 0.5*num.cos((deltaf*num.arange(hi(a), hi(b))-a)/(b-a)*num.pi)
    tap[hi(b):hi(c)] = 1.
    tap[hi(c):hi(d)] = 0.5 \
        + 0.5*num.cos((deltaf*num.arange(hi(c), hi(d))-c)/(d-c)*num.pi)

    return tap


def t2ind(t, tdelta, snap=round):
    return int(snap(t/tdelta))


def hilbert(x, N=None):
    '''
    Return the hilbert transform of x of length N.

    (from scipy.signal, but changed to use fft and ifft from numpy.fft)
    '''

    x = num.asarray(x)
    if N is None:
        N = len(x)
    if N <= 0:
        raise ValueError('N must be positive.')
    if num.iscomplexobj(x):
        logger.warning('imaginary part of x ignored.')
        x = num.real(x)

    Xf = num.fft.fft(x, N, axis=0)
    h = num.zeros(N)
    if N % 2 == 0:
        h[0] = h[N//2] = 1
        h[1:N//2] = 2
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    if len(x.shape) > 1:
        h = h[:, num.newaxis]
    x = num.fft.ifft(Xf*h)
    return x


def near(a, b, eps):
    return abs(a-b) < eps


def coroutine(func):
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    wrapper.__name__ = func.__name__
    wrapper.__dict__ = func.__dict__
    wrapper.__doc__ = func.__doc__
    return wrapper


class States(object):
    '''
    Utility to store channel-specific state in coroutines.
    '''

    def __init__(self):
        self._states = {}

    def get(self, tr):
        k = tr.codes
        if k in self._states:
            tmin, deltat, dtype, value = self._states[k]
            if (near(tmin, tr.tmin, deltat/100.)
                    and near(deltat, tr.deltat, deltat/10000.)
                    and dtype == tr.ydata.dtype):

                return value

        return None

    def set(self, tr, value, prevent_free=False):
        k = tr.codes
        if not prevent_free and k in self._states \
                and self._states[k][-1] is not value:

            self.free(self._states[k][-1])

        self._states[k] = (
            tr.tmax+tr.deltat, tr.deltat, tr.ydata.dtype, value)

    def clear(self, tr, prevent_free=False):
        k = tr.codes
        if not prevent_free and k in self._states:
            self.free(self._states[k][-1])

        del self._states[k]

    def clear_all(self):
        for k in self._states:
            self.free(self._states[k][-1])

        self._states = {}

    def free(self, value):
        pass

    def get_times(self):
        return dict(
            (codes, v[0]) for (codes, v) in self._states.items())


@coroutine
def co_list_append(list):
    while True:
        list.append((yield))


class ScipyBug(Exception):
    pass


@coroutine
def co_lfilter(target, b, a):
    '''
    Successively filter broken continuous trace data (coroutine).

    Create coroutine which takes :py:class:`Trace` objects, filters their data
    through :py:func:`scipy.signal.lfilter` and sends new :py:class:`Trace`
    objects containing the filtered data to target. This is useful, if one
    wants to filter a long continuous time series, which is split into many
    successive traces without producing filter artifacts at trace boundaries.

    Filter states are kept *per channel*, specifically, for each (network,
    station, location, channel) combination occuring in the input traces, a
    separate state is created and maintained. This makes it possible to filter
    multichannel or multistation data with only one :py:func:`co_lfilter`
    instance.

    Filter state is reset, when gaps occur.

    Use it like this::

      from pyrocko.trace import co_lfilter, co_list_append

      filtered_traces = []
      pipe = co_lfilter(co_list_append(filtered_traces), a, b)
      for trace in traces:
           pipe.send(trace)

      pipe.close()

    '''

    try:
        states = States()
        output = None
        while True:
            input = (yield)

            zi = states.get(input)
            if zi is None:
                zi = num.zeros(max(len(a), len(b))-1, dtype=float)

            output = input.copy(data=False)
            try:
                ydata, zf = signal.lfilter(b, a, input.get_ydata(), zi=zi)
            except ValueError:
                raise ScipyBug(
                    'signal.lfilter failed: could be related to a bug '
                    'in some older scipy versions, e.g. on opensuse42.1')

            output.set_ydata(ydata)
            states.set(input, zf)
            target.send(output)

    except GeneratorExit:
        target.close()


def co_antialias(target, q, n=None, ftype='fir'):
    b, a, n = util.decimate_coeffs(q, n, ftype)
    anti = co_lfilter(target, b, a)
    return anti


@coroutine
def co_dropsamples(target, q, nfir):
    try:
        states = States()
        while True:
            tr = (yield)
            newdeltat = q * tr.deltat
            ioffset = states.get(tr)
            if ioffset is None:
                # for fir filter, the first nfir samples are pulluted by
                # boundary effects; cut it off.
                # for iir this may be (much) more, we do not correct for that.
                # put sample instances to a time which is a multiple of the
                # new sampling interval.
                newtmin_want = math.ceil(
                    (tr.tmin+(nfir+1)*tr.deltat) / newdeltat) * newdeltat \
                    - (nfir/2*tr.deltat)
                ioffset = int(round((newtmin_want - tr.tmin)/tr.deltat))
                if ioffset < 0:
                    ioffset = ioffset % q

            newtmin_have = tr.tmin + ioffset * tr.deltat
            newtr = tr.copy(data=False)
            newtr.deltat = newdeltat
            # because the fir kernel shifts data by nfir/2 samples:
            newtr.tmin = newtmin_have - (nfir/2*tr.deltat)
            newtr.set_ydata(tr.get_ydata()[ioffset::q].copy())
            states.set(tr, (ioffset % q - tr.data_len() % q) % q)
            target.send(newtr)

    except GeneratorExit:
        target.close()


def co_downsample(target, q, n=None, ftype='fir'):
    '''
    Successively downsample broken continuous trace data (coroutine).

    Create coroutine which takes :py:class:`Trace` objects, downsamples their
    data and sends new :py:class:`Trace` objects containing the downsampled
    data to target.  This is useful, if one wants to downsample a long
    continuous time series, which is split into many successive traces without
    producing filter artifacts and gaps at trace boundaries.

    Filter states are kept *per channel*, specifically, for each (network,
    station, location, channel) combination occuring in the input traces, a
    separate state is created and maintained. This makes it possible to filter
    multichannel or multistation data with only one :py:func:`co_lfilter`
    instance.

    Filter state is reset, when gaps occur. The sampling instances are choosen
    so that they occur at (or as close as possible) to even multiples of the
    sampling interval of the downsampled trace (based on system time).
    '''

    b, a, n = util.decimate_coeffs(q, n, ftype)
    return co_antialias(co_dropsamples(target, q, n), q, n, ftype)


@coroutine
def co_downsample_to(target, deltat):

    decimators = {}
    try:
        while True:
            tr = (yield)
            ratio = deltat / tr.deltat
            rratio = round(ratio)
            if abs(rratio - ratio)/ratio > 0.0001:
                raise util.UnavailableDecimation('ratio = %g' % ratio)

            deci_seq = tuple(x for x in util.decitab(int(rratio)) if x != 1)
            if deci_seq not in decimators:
                pipe = target
                for q in deci_seq[::-1]:
                    pipe = co_downsample(pipe, q)

                decimators[deci_seq] = pipe

            decimators[deci_seq].send(tr)

    except GeneratorExit:
        for g in decimators.values():
            g.close()


class DomainChoice(StringChoice):
    choices = [
        'time_domain',
        'frequency_domain',
        'envelope',
        'absolute',
        'cc_max_norm']


class MisfitSetup(Object):
    '''
    Contains misfit setup to be used in :py:meth:`Trace.misfit`

    :param description: Description of the setup
    :param norm: L-norm classifier
    :param taper: Object of :py:class:`Taper`
    :param filter: Object of :py:class:`~pyrocko.response.FrequencyResponse`
    :param domain: ['time_domain', 'frequency_domain', 'envelope', 'absolute',
        'cc_max_norm']

    Can be dumped to a yaml file.
    '''

    xmltagname = 'misfitsetup'
    description = String.T(optional=True)
    norm = Int.T(optional=False)
    taper = Taper.T(optional=False)
    filter = FrequencyResponse.T(optional=True)
    domain = DomainChoice.T(default='time_domain')


def equalize_sampling_rates(trace_1, trace_2):
    '''
    Equalize sampling rates of two traces (reduce higher sampling rate to
    lower).

    :param trace_1: :py:class:`Trace` object
    :param trace_2: :py:class:`Trace` object

    Returns a copy of the resampled trace if resampling is needed.
    '''

    if same_sampling_rate(trace_1, trace_2):
        return trace_1, trace_2

    if trace_1.deltat < trace_2.deltat:
        t1_out = trace_1.copy()
        t1_out.downsample_to(deltat=trace_2.deltat, snap=True)
        logger.debug('Trace downsampled (return copy of trace): %s'
                     % '.'.join(t1_out.nslc_id))
        return t1_out, trace_2

    elif trace_1.deltat > trace_2.deltat:
        t2_out = trace_2.copy()
        t2_out.downsample_to(deltat=trace_1.deltat, snap=True)
        logger.debug('Trace downsampled (return copy of trace): %s'
                     % '.'.join(t2_out.nslc_id))
        return trace_1, t2_out


def Lx_norm(u, v, norm=2):
    '''
    Calculate the misfit denominator *m* and the normalization divisor *n*
    according to norm.

    The normalization divisor *n* is calculated from ``v``.

    :param u: :py:class:`numpy.ndarray`
    :param v: :py:class:`numpy.ndarray`
    :param norm: (default = 2)

    ``u`` and ``v`` must be of same size.
    '''

    if norm == 1:
        return (
            num.sum(num.abs(v-u)),
            num.sum(num.abs(v)))

    elif norm == 2:
        return (
            num.sqrt(num.sum((v-u)**2)),
            num.sqrt(num.sum(v**2)))

    else:
        return (
            num.power(num.sum(num.abs(num.power(v - u, norm))), 1./norm),
            num.power(num.sum(num.abs(num.power(v, norm))), 1./norm))


def do_downsample(tr, deltat):
    if abs(tr.deltat - deltat) / tr.deltat > 1e-6:
        tr = tr.copy()
        tr.downsample_to(deltat, snap=True, demean=False)
    else:
        if tr.tmin/tr.deltat > 1e-6 or tr.tmax/tr.deltat > 1e-6:
            tr = tr.copy()
            tr.snap()
    return tr


def do_extend(tr, tmin, tmax):
    if tmin < tr.tmin or tmax > tr.tmax:
        tr = tr.copy()
        tr.extend(tmin=tmin, tmax=tmax, fillmethod='repeat')

    return tr


def do_pre_taper(tr, taper):
    return tr.taper(taper, inplace=False, chop=True)


def do_fft(tr, filter):
    if filter is None:
        return tr
    else:
        ndata = tr.ydata.size
        nfft = nextpow2(ndata)
        padded = num.zeros(nfft, dtype=float)
        padded[:ndata] = tr.ydata
        spectrum = num.fft.rfft(padded)
        df = 1.0 / (tr.deltat * nfft)
        frequencies = num.arange(spectrum.size)*df
        return [tr, frequencies, spectrum]


def do_filter(inp, filter):
    if filter is None:
        return inp
    else:
        tr, frequencies, spectrum = inp
        spectrum *= filter.evaluate(frequencies)
        return [tr, frequencies, spectrum]


def do_ifft(inp):
    if isinstance(inp, Trace):
        return inp
    else:
        tr, _, spectrum = inp
        ndata = tr.ydata.size
        tr = tr.copy(data=False)
        tr.set_ydata(num.fft.irfft(spectrum)[:ndata])
        return tr


def check_alignment(t1, t2):
    if abs(t1.tmin-t2.tmin) > t1.deltat * 1e-4 or \
            abs(t1.tmax - t2.tmax) > t1.deltat * 1e-4 or \
            t1.ydata.shape != t2.ydata.shape:
        raise MisalignedTraces(
            'Cannot calculate misfit of %s and %s due to misaligned '
            'traces.' % ('.'.join(t1.nslc_id), '.'.join(t2.nslc_id)))


def check_overlaps(traces_a, traces_b=None, message='Traces overlap.'):
    for ia, tr_a in enumerate(traces_a):
        for tr_b in traces_a[ia+1:] if traces_b is None else traces_b:
            if tr_a.nslc_id == tr_b.nslc_id and tr_a.overlaps(
                    tr_b.tmin, tr_b.tmax):
                raise OverlappingTraces(
                    message + '\n  Trace 1: %s\n  Trace 2: %s' % (
                        tr_a.summary, tr_b.summary))
