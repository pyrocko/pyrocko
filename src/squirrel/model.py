# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Data model and content types handled by the Squirrel framework.

Squirrel uses flat content types to represent waveform, station, channel,
response, event, and a few other objects. A common subset of the information in
these objects is indexed in the database, currently: kind, codes, time interval
and sampling rate. The :py:class:`Nut` objects encapsulate this information
together with information about where and how to get the associated bulk data.

Further content types are defined here to handle waveform orders, waveform
promises, data coverage and sensor information.
'''

import math
import hashlib
from os import urandom
from base64 import urlsafe_b64encode
from collections import defaultdict

import numpy as num

from pyrocko import util
from pyrocko.guts import Object, String, Timestamp, Float, Int, \
    Unicode, Tuple, List, StringChoice, Any, Dict, Duration, clone
from pyrocko.model import squirrel_content, Location
from pyrocko.response import FrequencyResponse, MultiplyResponse, \
    IntegrationResponse, DifferentiationResponse, simplify_responses, \
    FrequencyResponseCheckpoint, Gain
from pyrocko.model.codes import CodesError, Codes, CodesNSLCE, CodesNSL, \
    CodesX, CodesMatcher, match_codes, match_codes_any, classify_patterns, \
    merge_codes # noqa

from .error import ConversionError, SensorAggregationError


d2r = num.pi / 180.
r2d = 1.0 / d2r


guts_prefix = 'squirrel'


def mkvec(x, y, z):
    return num.array([x, y, z], dtype=float)


def are_orthogonal(vecs, eps=0.01):
    return all(abs(
        num.dot(vecs[i], vecs[j]) < eps
        for (i, j) in [(0, 1), (1, 2), (2, 0)]))


g_content_kinds = [
    'undefined',
    'waveform',
    'station',
    'channel',
    'response',
    'event',
    'waveform_promise',
    'empty',
    'carpet',
    'elk',
]

g_codes_classes = [
    CodesX,
    CodesNSLCE,
    CodesNSL,
    CodesNSLCE,
    CodesNSLCE,
    CodesX,
    CodesNSLCE,
    CodesX,
    CodesNSLCE,
    CodesX]

g_codes_classes_ndot = {
    0: CodesX,
    2: CodesNSL,
    3: CodesNSLCE,
    4: CodesNSLCE}


def to_codes_simple(kind_id, codes_safe_str):
    '''
    Convert a safe-string representation of codes to a codes object.

    :param kind_id:
        Content kind determining the target codes class, one of the ids
        given in :py:data:`g_content_kind_ids`.
    :type kind_id:
        int

    :param codes_safe_str:
        Codes as a database-safe string representation.
    :type codes_safe_str:
        str

    :rtype:
        :py:class:`~pyrocko.model.codes.Codes` (subclass appropriate for
        ``kind_id``)
    '''
    return g_codes_classes[kind_id](safe_str=codes_safe_str)


def to_codes(kind_id, obj):
    '''
    Convert an object to a codes object of the class appropriate for a
    given content kind.

    :param kind_id:
        Content kind determining the target codes class, one of the ids
        given in :py:data:`g_content_kind_ids`.
    :type kind_id:
        int

    :param obj:
        Value to be converted, anything accepted by the target codes
        class's constructor (e.g. a dot-separated string or a tuple).

    :rtype:
        :py:class:`~pyrocko.model.codes.Codes` (subclass appropriate for
        ``kind_id``)
    '''
    return g_codes_classes[kind_id](obj)


def to_codes_guess(s):
    '''
    Convert a dot-separated string to a codes object, guessing the
    appropriate codes class from its number of dots.

    ``STA`` (no dots) gives :py:class:`~pyrocko.model.codes.CodesX`, which
    matches the station code when used for channel or station queries, and
    the name for other content kinds. ``NET.STA`` gives
    :py:class:`~pyrocko.model.codes.CodesNSL` with any location code.

    :param s:
        Dot-separated codes string, e.g. ``'NET.STA.LOC.CHA'``.
    :type s:
        str

    :rtype:
        :py:class:`~pyrocko.model.codes.Codes`

    :raises CodesError:
        If the number of dots does not match any known codes class.
    '''
    ndot = s.count('.')
    if ndot == 1:
        # NET.STA, any location
        return CodesNSL(s + '.*')

    try:
        return g_codes_classes_ndot[ndot](s)
    except KeyError:
        raise CodesError('Cannot guess codes type: %s' % s)


# derived list class to enable detection of already preprocessed codes patterns
class codes_patterns_list(list):
    pass


def codes_patterns_for_kind(kind_id, codes):
    '''
    Convert codes patterns to a normalized list of query patterns.

    :param kind_id:
        Content kind determining the target codes class, one of the ids
        given in :py:data:`g_content_kind_ids`.
    :type kind_id:
        int

    :param codes:
        Single codes pattern, or list thereof, in any form accepted by
        :py:func:`to_codes`. Already-normalized input (as returned by this
        function) is passed through unchanged.
    :type codes:
        anything accepted by :py:func:`to_codes`, or
        :py:class:`list` thereof

    :returns:
        Normalized list of codes patterns, appropriate for the queried
        content kind. For station queries, an additional pattern matching
        any location code is added, since SEED/StationXML conventions
        treat the location code as part of the channel rather than the
        station, unlike :py:class:`~pyrocko.model.codes.CodesNSL` (see its
        docstring).
    :rtype:
        :py:class:`codes_patterns_list`
    '''
    if isinstance(codes, codes_patterns_list):
        return codes

    if isinstance(codes, list):
        lcodes = codes_patterns_list()
        for sc in codes:
            lcodes.extend(codes_patterns_for_kind(kind_id, sc))

        return lcodes

    codes = to_codes(kind_id, codes)

    lcodes = codes_patterns_list()
    lcodes.append(codes)
    if kind_id == STATION:
        lcodes.append(codes.replace(location='[*]'))

    return lcodes


g_content_kind_ids = (
    UNDEFINED, WAVEFORM, STATION, CHANNEL, RESPONSE, EVENT,
    WAVEFORM_PROMISE, EMPTY, CARPET, ELK) = range(len(g_content_kinds))


g_tmin, g_tmax = util.get_working_system_time_range()[:2]


try:
    g_tmin_queries = max(g_tmin, util.str_to_time_fillup('1900-01-01'))
except Exception:
    g_tmin_queries = g_tmin


def to_kind(kind_id):
    '''
    Get content kind name for a content kind id.

    :param kind_id:
        Content kind id, one of the ids given in
        :py:data:`g_content_kind_ids`.
    :type kind_id:
        int

    :rtype:
        str
    '''
    return g_content_kinds[kind_id]


def to_kinds(kind_ids):
    '''
    Get content kind names for a list of content kind ids, see
    :py:func:`to_kind`.
    '''
    return [g_content_kinds[kind_id] for kind_id in kind_ids]


def to_kind_id(kind):
    '''
    Get content kind id for a content kind name.

    :param kind:
        Content kind name, e.g. ``'waveform'``, ``'station'``,
        ``'channel'``, ``'response'`` or ``'event'``.
    :type kind:
        str

    :rtype:
        int
    '''
    return g_content_kinds.index(kind)


def to_kind_ids(kinds):
    '''
    Get content kind ids for a list of content kind names, see
    :py:func:`to_kind_id`.
    '''
    return [g_content_kinds.index(kind) for kind in kinds]


g_kind_mask_all = 0xffff


def to_kind_mask(kinds):
    if kinds:
        return sum(1 << kind_id for kind_id in to_kind_ids(kinds))
    else:
        return g_kind_mask_all


def str_or_none(x):
    if x is None:
        return None
    else:
        return str(x)


def float_or_none(x):
    if x is None:
        return None
    else:
        return float(x)


def int_or_none(x):
    if x is None:
        return None
    else:
        return int(x)


def int_or_g_tmin(x):
    if x is None:
        return g_tmin
    else:
        return int(x)


def int_or_g_tmax(x):
    if x is None:
        return g_tmax
    else:
        return int(x)


def tmin_or_none(x):
    if x == g_tmin:
        return None
    else:
        return x


def tmax_or_none(x):
    if x == g_tmax:
        return None
    else:
        return x


def time_or_none_to_str(x):
    if x is None:
        return '...'.ljust(17)
    else:
        return util.time_to_str(x)


def codes_to_str_abbreviated(codes, indent='  '):
    codes = [str(x) for x in codes]

    if len(codes) > 20:
        scodes = '\n' + util.ewrap(codes[:10], indent=indent) \
            + '\n%s[%i more]\n' % (indent, len(codes) - 20) \
            + util.ewrap(codes[-10:], indent='  ')
    else:
        scodes = '\n' + util.ewrap(codes, indent=indent) \
            if codes else '<none>'

    return scodes


g_offset_time_unit_inv = 1000000000
g_offset_time_unit = 1.0 / g_offset_time_unit_inv


def tsplit(t):
    '''
    Split time into full-seconds and fractional-offset parts, for compact,
    fast-comparable database storage.

    :param t:
        Time to be split, or ``None``.
    :type t:
        :py:func:`~pyrocko.util.get_time_float`

    :returns:
        ``(seconds, offset)``, where ``seconds`` is the floored full
        seconds part and ``offset`` the remaining fractional part, in
        integer units of :py:data:`g_offset_time_unit` seconds. Returns
        ``(None, 0)`` if ``t`` is ``None``.
    :rtype:
        :py:class:`tuple` of (:py:class:`int` or ``None``, :py:class:`int`)
    '''
    if t is None:
        return None, 0

    t = util.to_time_float(t)
    if type(t) is float:
        t = round(t, 5)
    else:
        t = round(t, 9)

    seconds = num.floor(t)
    offset = t - seconds
    return int(seconds), int(round(offset * g_offset_time_unit_inv))


def tjoin(seconds, offset):
    '''
    Join a full-seconds/fractional-offset pair back into a single time
    value (inverse of :py:func:`tsplit`).

    :param seconds:
        Full seconds part, or ``None``.
    :type seconds:
        int

    :param offset:
        Fractional part, in integer units of :py:data:`g_offset_time_unit`
        seconds.
    :type offset:
        int

    :returns:
        Joined time, or ``None`` if ``seconds`` is ``None``.
    :rtype:
        :py:func:`~pyrocko.util.get_time_float`
    '''
    if seconds is None:
        return None

    return util.to_time_float(seconds) \
        + util.to_time_float(offset*g_offset_time_unit)


tscale_min = 1
tscale_max = 365 * 24 * 3600  # last edge is one above
tscale_logbase = 20

tscale_edges = [tscale_min]
while True:
    tscale_edges.append(tscale_edges[-1]*tscale_logbase)
    if tscale_edges[-1] >= tscale_max:
        break


tscale_edges = num.array(tscale_edges)


def tscale_to_kscale(tscale):

    # 0 <= x < tscale_edges[1]: 0
    # tscale_edges[1] <= x < tscale_edges[2]: 1
    # ...
    # tscale_edges[len(tscale_edges)-1] <= x: len(tscale_edges)

    return int(num.searchsorted(tscale_edges, tscale))


def get_selection_args(
        kind_id, obj=None, tmin=None, tmax=None, time=None, codes=None):

    if codes is not None:
        codes = codes_patterns_for_kind(kind_id, codes)

    if time is not None:
        tmin = time
        tmax = time

    if obj is not None:
        tmin = tmin if tmin is not None else obj.tmin
        tmax = tmax if tmax is not None else obj.tmax
        codes = codes if codes is not None else codes_patterns_for_kind(
            kind_id, obj.codes)

    return tmin, tmax, codes


@squirrel_content
class Station(Location):
    '''
    A seismic station.
    '''

    codes = CodesNSL.T(
        help='Codes identifying the station (NET.STA.LOC).')

    tmin = Timestamp.T(
        optional=True,
        help='Start time of station epoch.')
    tmax = Timestamp.T(
        optional=True,
        help='End time of station epoch.')

    description = Unicode.T(
        optional=True,
        help='Additional description of the station.')

    def __init__(self, **kwargs):
        kwargs['codes'] = CodesNSL(kwargs['codes'])
        Location.__init__(self, **kwargs)

    @property
    def time_span(self):
        '''
        Get station's epoch as ``(tmin, tmax)`` tuple.
        '''
        return (self.tmin, self.tmax)

    def get_pyrocko_station(self):
        '''
        Get station as a classic Pyrocko station object.

        :returns:
            Converted station object.
        :rtype:
            :py:class:`pyrocko.model.station.Station`
        '''
        from pyrocko import model
        return model.Station(*self._get_pyrocko_station_args())

    def _get_pyrocko_station_args(self):
        return (
            self.codes.network,
            self.codes.station,
            self.codes.location,
            self.lat,
            self.lon,
            self.elevation,
            self.depth,
            self.north_shift,
            self.east_shift)


@squirrel_content
class ChannelBase(Location):
    '''
    Base class for :py:class:`Channel` and :py:class:`Sensor`, providing
    the common properties of a single seismic recording channel or a group
    of channels forming a sensor.
    '''

    codes = CodesNSLCE.T(
        help='Codes identifying the channel (NET.STA.LOC.CHA.EXTRA).')

    tmin = Timestamp.T(
        optional=True,
        help='Start time of channel epoch.')
    tmax = Timestamp.T(
        optional=True,
        help='End time of channel epoch.')

    deltat = Float.T(
        optional=True,
        help='Sampling interval [s].')

    def __init__(self, **kwargs):
        kwargs['codes'] = CodesNSLCE(kwargs['codes'])
        Location.__init__(self, **kwargs)

    @property
    def time_span(self):
        '''
        Get channel's epoch as ``(tmin, tmax)`` tuple.
        '''
        return (self.tmin, self.tmax)

    def overlaps(self, tmin, tmax):
        '''
        Check if channel's epoch overlaps with a given time span.

        :param tmin:
            Start time of query interval.
        :type tmin:
            :py:func:`~pyrocko.util.get_time_float`

        :param tmax:
            End time of query interval.
        :type tmax:
            :py:func:`~pyrocko.util.get_time_float`

        :returns:
            ``True`` if the channel's epoch (:py:attr:`tmin`,
            :py:attr:`tmax`) has any overlap with the given interval.
        :rtype:
            bool
        '''
        return (
            self.tmin is None or tmax is None
            or self.tmin <= tmax
        ) and (
            tmin is None or self.tmax is None
            or tmin < self.tmax
        )

    def _get_sensor_codes(self):
        return self.codes.replace(
            channel=self.codes.channel[:-1] + '?')

    def _get_sensor_args(self):
        def getattr_rep(k):
            if k == 'codes':
                return self._get_sensor_codes()
            else:
                return getattr(self, k)

        return tuple(getattr_rep(k) for k in ChannelBase.T.propnames)

    def _get_channel_args(self, component):
        def getattr_rep(k):
            if k == 'codes':
                return self.codes.replace(
                    channel=self.codes.channel[:-1] + component)
            else:
                return getattr(self, k)

        return tuple(getattr_rep(k) for k in ChannelBase.T.propnames)

    def _get_pyrocko_station_args(self):
        return (
            self.codes.network,
            self.codes.station,
            self.codes.location,
            self.lat,
            self.lon,
            self.elevation,
            self.depth,
            self.north_shift,
            self.east_shift)


class Channel(ChannelBase):
    '''
    A channel of a seismic station.
    '''

    dip = Float.T(
        optional=True,
        help='Dip angle of channel [deg], -90 is up, 0 is horizontal, '
             '90 is down.')
    azimuth = Float.T(
        optional=True,
        help='Azimuth angle of channel [deg], 0 is north, 90 is east.')

    def get_pyrocko_channel(self):
        '''
        Get channel as a classic Pyrocko channel object.

        :returns:
            Converted channel object.
        :rtype:
            :py:class:`pyrocko.model.station.Channel`
        '''
        from pyrocko import model
        return model.Channel(*self._get_pyrocko_channel_args())

    def _get_pyrocko_channel_args(self):
        return (
            self.codes.channel,
            self.azimuth,
            self.dip)

    @property
    def orientation_enz(self):
        '''
        Get sensor orientation as unit vector in East-North-Up coordinates,
        or ``None`` if azimuth or dip are not set.
        '''
        if None in (self.azimuth, self.dip):
            return None

        n = math.cos(self.azimuth*d2r)*math.cos(self.dip*d2r)
        e = math.sin(self.azimuth*d2r)*math.cos(self.dip*d2r)
        d = math.sin(self.dip*d2r)
        return mkvec(e, n, -d)


def cut_intervals(channels):
    channels = list(channels)
    times = set()
    for channel in channels:
        if channel.tmin is not None:
            times.add(channel.tmin)
        if channel.tmax is not None:
            times.add(channel.tmax)

    times = sorted(times)

    if any(channel.tmin is None for channel in channels):
        times[0:0] = [None]

    if any(channel.tmax is None for channel in channels):
        times.append(None)

    if len(times) <= 2:
        return channels

    channels_out = []
    for channel in channels:
        for i in range(len(times)-1):
            tmin = times[i]
            tmax = times[i+1]
            if ((channel.tmin is None or (
                    tmin is not None and channel.tmin <= tmin))
                    and (channel.tmax is None or (
                        tmax is not None and tmax <= channel.tmax))):

                channel_out = clone(channel)
                channel_out.tmin = tmin
                channel_out.tmax = tmax
                channels_out.append(channel_out)

    return channels_out


class ProjectionError(Exception):
    pass


g_default_earthmodel = None


def get_default_earthmodel():
    from pyrocko import cake
    global g_default_earthmodel
    if g_default_earthmodel is None:
        g_default_earthmodel = cake.load_model()

    return g_default_earthmodel


class Sensor(ChannelBase):
    '''
    Representation of a channel group.
    '''

    channels = List.T(
        Channel.T(),
        help='Channels forming this sensor, e.g. the three components of a '
             'three-component seismometer.')

    @classmethod
    def from_channels(cls, channels):
        '''
        Group channels into sensors.

        :param channels:
            Channels to be grouped.
        :type channels:
            iterable of :py:class:`Channel` objects

        :returns:
            Sensors formed by grouping the given channels by matching
            station/location codes, orientation-independent channel code
            (i.e. NET.STA.LOC.BI, where B and I are the SEED band and
            instrument codes, e.g. ``'HH'``, omitting the component
            character) and time span, splitting groups where the
            individual channels' time spans don't fully align.
        :rtype:
            :py:class:`list` of :py:class:`Sensor` objects
        '''
        groups = defaultdict(list)
        for channel in channels:
            groups[channel._get_sensor_codes()].append(channel)

        channels_cut = []
        for group in groups.values():
            channels_cut.extend(cut_intervals(group))

        groups = defaultdict(list)
        for channel in channels_cut:
            groups[channel._get_sensor_args()].append(channel)

        return [
            cls(channels=channels,
                **dict(zip(ChannelBase.T.propnames, args)))
            for args, channels in groups.items()]

    @classmethod
    def from_channels_single(cls, channels):
        args = channels[0]._get_sensor_args()
        for channel in channels[1:]:
            if args != channel._get_sensor_args():
                raise SensorAggregationError(
                    'Cannot create sensor from incompatible channels:'
                    '\n  %s' % (
                        '\n  '.join(channel.summary for channel in channels)))

        return cls(
            channels=channels,
            **dict(zip(ChannelBase.T.propnames, args)))

    def channel_vectors(self):
        '''
        Get orientation unit vectors of member channels.

        :returns:
            Orientation vectors of the member channels in East-North-Up
            coordinates, stacked into a ``(nchannels, 3)`` array, in the
            order of :py:attr:`channels`.
        :rtype:
            :py:class:`numpy.ndarray`
        '''
        return num.vstack(
            [channel.orientation_enz for channel in self.channels])

    def projected_channels(self, system, component_names):
        '''
        Get synthetic channel definitions for a given target coordinate
        system.

        :param system:
            Target coordinate system as a ``(3, 3)`` rotation matrix, rows
            giving the new component directions in East-North-Up
            coordinates.
        :type system:
            :py:class:`numpy.ndarray`

        :param component_names:
            Component codes to be used for the three output channels, e.g.
            ``'ENZ'``.
        :type component_names:
            :py:class:`str` or :py:class:`list` of :py:class:`str`

        :returns:
            Synthetic channel definitions for the projected components.
        :rtype:
            :py:class:`list` of :py:class:`Channel` objects
        '''
        return [
            Channel(
                azimuth=math.atan2(e, n) * r2d,
                dip=-math.asin(z) * r2d,
                **dict(zip(
                    ChannelBase.T.propnames,
                    self._get_channel_args(comp))))
            for comp, (e, n, z) in zip(component_names, system)]

    def matrix_to(self, system, epsilon=1e-7):
        '''
        Get rotation matrix from member channels to a given coordinate
        system.

        :param system:
            Target coordinate system as a ``(3, 3)`` rotation matrix, rows
            giving the new component directions in East-North-Up
            coordinates.
        :type system:
            :py:class:`numpy.ndarray`

        :param epsilon:
            Absolute values below this threshold are set to zero in the
            returned matrix, to clean up numerical noise.
        :type epsilon:
            float

        :returns:
            Rotation matrix mapping from the member channels' orientations
            to the given target system.
        :rtype:
            :py:class:`numpy.ndarray`
        '''
        m = num.dot(system, self.channel_vectors().T)
        m[num.abs(m) < epsilon] = 0.0
        return m

    def projection_to(self, system, component_names):
        '''
        Get projection to a given coordinate system.

        :param system:
            Target coordinate system as a ``(3, 3)`` rotation matrix, rows
            giving the new component directions in East-North-Up
            coordinates.
        :type system:
            :py:class:`numpy.ndarray`

        :param component_names:
            Component codes to be used for the three output channels, e.g.
            ``'ENZ'``.
        :type component_names:
            :py:class:`str` or :py:class:`list` of :py:class:`str`

        :returns:
            A tuple ``(matrix, in_channels, out_channels)`` as expected by
            :py:func:`pyrocko.trace.project`.
        :rtype:
            :py:class:`tuple`
        '''
        return (
            self.matrix_to(system),
            self.channels,
            self.projected_channels(system, component_names))

    def projection_to_enz(self):
        '''
        Get projection to East-North-Up coordinates.

        :returns:
            A tuple ``(matrix, in_channels, out_channels)`` as expected by
            :py:func:`pyrocko.trace.project`.
        :rtype:
            :py:class:`tuple`
        '''
        return self.projection_to(num.identity(3), 'ENZ')

    def projection_to_trz(self, source, azimuth=None):
        '''
        Get projection to Transverse-Radial-Up coordinates.

        :param source:
            Source location to compute the backazimuth from, or ``None``
            if ``azimuth`` is given directly.
        :type source:
            :py:class:`~pyrocko.model.location.Location`

        :param azimuth:
            Azimuth of the sensor as seen from the source [deg], overriding
            the automatic backazimuth computation from ``source``.
        :type azimuth:
            float

        :returns:
            A tuple ``(matrix, in_channels, out_channels)`` as expected by
            :py:func:`pyrocko.trace.project`.
        :rtype:
            :py:class:`tuple`
        '''
        if azimuth is not None:
            assert source is None
        else:
            azimuth = source.azibazi_to(self)[1] + 180.

        return self.projection_to(num.array([
            [math.cos(azimuth*d2r), -math.sin(azimuth*d2r), 0.],
            [math.sin(azimuth*d2r), math.cos(azimuth*d2r), 0.],
            [0., 0., 1.]], dtype=float), 'TRZ')

    def incidence_angle(
            self,
            source=None,
            earthmodel=None,
            phases=None,
            distance=None,
            source_depth=None):

        from pyrocko import cake
        if earthmodel is None:
            earthmodel = get_default_earthmodel()

        if len(phases) == 0:
            phases = [cake.PhaseDef(x) for x in ['p', 'P']]

        if distance is None:
            if source is None:
                raise ProjectionError(
                    'Cannot determine incidence angle: '
                    'Neither `source` nor `distance` given.')

            distance = source.distance_to(self) * cake.m2d

        if source_depth is None:
            if source is None:
                raise ProjectionError(
                    'Cannot determine incidence angle: '
                    'Neither `source` nor `source_depth` given.')

            source_depth = source.depth

        rays = earthmodel.arrivals(
            [distance],
            phases,
            zstart=source_depth,
            zstop=self.depth)

        if len(rays) == 0:
            raise ProjectionError(
                'Cannot determine incidence angle: No rays connecting '
                'source and sensor could be determined.')

        return rays[0].incidence_angle()

    def projection_to_lqt(
            self,
            source=None,
            earthmodel=None,
            phases=None,
            distance=None,
            source_depth=None,
            azimuth=None,
            incidence=None):

        if azimuth is None:
            if source is None:
                raise ProjectionError(
                    'Cannot determine azimuth angle: No `source` given.')

            azimuth = source.azibazi_to(self)[1]

        if incidence is None:
            incidence = self.incidence_angle(
                source=source,
                earthmodel=earthmodel,
                phases=phases,
                distance=distance,
                source_depth=source_depth)

        # reference is Plesinger et al., 1986
        iprime = incidence
        ca = math.cos(azimuth*d2r)
        sa = math.sin(azimuth*d2r)
        ci = math.cos(iprime*d2r)
        si = math.sin(iprime*d2r)

        return self.projection_to(num.array([
            [-si*sa, -si*ca, ci],  # noqa
            [ ci*sa,  ci*ca, si],  # noqa
            [   -ca,     sa, 0.]], dtype=float), 'LQT')  # noqa

    def project_to_enz(self, traces):
        '''
        Rotate traces to East-North-Up coordinates.

        :param traces:
            Traces to be rotated, matching this sensor's member channels.
        :type traces:
            :py:class:`list` of :py:class:`~pyrocko.trace.Trace` objects

        :returns:
            Rotated traces.
        :rtype:
            :py:class:`list` of :py:class:`~pyrocko.trace.Trace` objects
        '''
        from pyrocko import trace

        matrix, in_channels, out_channels = self.projection_to_enz()

        return trace.project(traces, matrix, in_channels, out_channels)

    def project_to_trz(self, source, traces, azimuth=None):
        '''
        Rotate traces to Transverse-Radial-Up coordinates.

        :param source:
            Source location to compute the backazimuth from, or ``None``
            if ``azimuth`` is given directly.
        :type source:
            :py:class:`~pyrocko.model.location.Location`

        :param traces:
            Traces to be rotated, matching this sensor's member channels.
        :type traces:
            :py:class:`list` of :py:class:`~pyrocko.trace.Trace` objects

        :param azimuth:
            Azimuth of the sensor as seen from the source [deg], overriding
            the automatic backazimuth computation from ``source``.
        :type azimuth:
            float

        :returns:
            Rotated traces.
        :rtype:
            :py:class:`list` of :py:class:`~pyrocko.trace.Trace` objects
        '''
        from pyrocko import trace

        matrix, in_channels, out_channels = self.projection_to_trz(
            source, azimuth=azimuth)

        return trace.project(traces, matrix, in_channels, out_channels)

    def project_to_lqt(
            self,
            source,
            traces,
            earthmodel=None,
            phases=None,
            distance=None,
            source_depth=None,
            azimuth=None,
            incidence=None):

        from pyrocko import trace

        matrix, in_channels, out_channels = self.projection_to_lqt(
            source=source,
            earthmodel=earthmodel,
            phases=phases,
            distance=distance,
            source_depth=source_depth,
            azimuth=azimuth,
            incidence=incidence)

        return trace.project(traces, matrix, in_channels, out_channels)


observational_quantities = [
    'acceleration', 'velocity', 'displacement', 'pressure',
    'rotation_displacement', 'rotation_velocity', 'rotation_acceleration',
    'temperature']


technical_quantities = [
    'voltage', 'counts']


class QuantityType(StringChoice):
    '''
    Choice of observational or technical quantity.

    SI units are used for all quantities, where applicable.
    '''
    choices = observational_quantities + technical_quantities


class ResponseStage(Object):
    '''
    Representation of a response stage.

    Components of a seismic recording system are represented as a sequence of
    response stages, e.g. sensor, pre-amplifier, digitizer, digital
    downsampling.
    '''
    input_quantity = QuantityType.T(
        optional=True,
        help='Physical quantity accepted at input of stage.')
    input_sample_rate = Float.T(
        optional=True,
        help='Sampling rate at input of stage [Hz].')
    output_quantity = QuantityType.T(
        optional=True,
        help='Physical quantity produced at output of stage.')
    output_sample_rate = Float.T(
        optional=True,
        help='Sampling rate at output of stage [Hz].')
    elements = List.T(
        FrequencyResponse.T(),
        help='Sequence of frequency responses combined (multiplied) to '
             'form this stage.')
    log = List.T(
        Tuple.T(3, String.T()),
        help='Log messages noting problems encountered or simplifications '
             'made while converting response information from its source '
             'format, with entries of the form ``(level, context, '
             'message)``.')

    @property
    def stage_type(self):
        '''
        Attempt to classify the stage as ``'sensor'``, ``'electronics'``,
        ``'digitizer'``, ``'decimation'``, ``'conversion'``,
        ``'combination'`` or ``'unknown'``, based on its input/output
        quantities and decimation factor.
        '''
        if self.input_quantity in observational_quantities \
                and self.output_quantity in observational_quantities:
            return 'conversion'

        if self.input_quantity in observational_quantities \
                and self.output_quantity == 'voltage':
            return 'sensor'

        elif self.input_quantity == 'voltage' \
                and self.output_quantity == 'voltage':
            return 'electronics'

        elif self.input_quantity == 'voltage' \
                and self.output_quantity == 'counts':
            return 'digitizer'

        elif self.decimation_factor is not None \
                and (self.input_quantity is None or self.input_quantity == 'counts') \
                and (self.output_quantity is None or self.output_quantity == 'counts'):  # noqa
            return 'decimation'

        elif self.input_quantity in observational_quantities \
                and self.output_quantity == 'counts':
            return 'combination'

        else:
            return 'unknown'

    @property
    def decimation_factor(self):
        '''
        Get decimation factor of stage, or ``None`` if it cannot be
        determined from :py:attr:`input_sample_rate` and
        :py:attr:`output_sample_rate`.
        '''
        irate = self.input_sample_rate
        orate = self.output_sample_rate
        if irate is not None and orate is not None \
                and irate > orate and irate / orate > 1.0:

            return irate / orate
        else:
            return None

    @property
    def summary_quantities(self):
        '''
        One-line summary of input/output quantities.
        '''
        return '%s -> %s' % (
            self.input_quantity or '?',
            self.output_quantity or '?')

    @property
    def summary_rates(self):
        '''
        One-line summary of input/output sampling rates.
        '''
        irate = self.input_sample_rate
        orate = self.output_sample_rate
        factor = self.decimation_factor

        if irate and orate is None:
            return '%g Hz' % irate

        elif orate and irate is None:
            return '%g Hz' % orate

        elif irate and orate and irate == orate:
            return '%g Hz' % irate

        elif any(x for x in (irate, orate, factor)):
            return '%s -> %s Hz (%s)' % (
                '%g' % irate if irate else '?',
                '%g' % orate if orate else '?',
                ':%g' % factor if factor else '?')
        else:
            return ''

    @property
    def summary_elements(self):
        '''
        One-line summary of the combined :py:attr:`elements`.
        '''
        xs = [x.summary for x in self.elements]
        return '%s' % ('*'.join(x for x in xs if x != 'one') or 'one')

    @property
    def summary_log(self):
        '''
        One-letter-per-entry summary of :py:attr:`log` severities.
        '''
        return ''.join(sorted(set(x[0][0].upper() for x in self.log)))

    @property
    def summary_entries(self):
        '''
        Get tuple of one-line summary values, for use with
        :py:func:`~pyrocko.util.fmt_summary`.
        '''
        return (
            self.__class__.__name__,
            self.stage_type,
            self.summary_log,
            self.summary_quantities,
            self.summary_rates,
            self.summary_elements)

    @property
    def summary(self):
        '''
        One-line summary of the response stage.
        '''
        return util.fmt_summary(self.summary_entries, (10, 15, 3, 30, 30, 0))

    def get_effective(self):
        '''
        Get effective frequency response of this stage.

        :returns:
            Combined response of all :py:attr:`elements`.
        :rtype:
            :py:class:`~pyrocko.response.MultiplyResponse`
        '''
        return MultiplyResponse(responses=list(self.elements))


D = 'displacement'
V = 'velocity'
A = 'acceleration'

g_converters = {
    (V, D): IntegrationResponse(1),
    (A, D): IntegrationResponse(2),
    (D, V): DifferentiationResponse(1),
    (A, V): IntegrationResponse(1),
    (D, A): DifferentiationResponse(2),
    (V, A): DifferentiationResponse(1)}


def response_converters(input_quantity, output_quantity):
    if input_quantity is None or input_quantity == output_quantity:
        return []

    if output_quantity is None:
        raise ConversionError('Unspecified target quantity.')

    try:
        return [g_converters[input_quantity, output_quantity]]

    except KeyError:
        raise ConversionError('No rule to convert from "%s" to "%s".' % (
            input_quantity, output_quantity))


@squirrel_content
class Response(Object):
    '''
    The instrument response of a seismic station channel.
    '''

    codes = CodesNSLCE.T(
        help='Codes identifying the channel (NET.STA.LOC.CHA.EXTRA) this '
             'response applies to.')
    tmin = Timestamp.T(
        optional=True,
        help='Start time of response epoch.')
    tmax = Timestamp.T(
        optional=True,
        help='End time of response epoch.')

    stages = List.T(
        ResponseStage.T(),
        help='Sequence of response stages, from sensor to digitizer '
             'output.')
    checkpoints = List.T(
        FrequencyResponseCheckpoint.T(),
        help='Gain checkpoints to be honoured, e.g. from a dataless SEED '
             'volume or a StationXML file, used to sanity-check the '
             'combined response computed from :py:attr:`stages`.')

    deltat = Float.T(
        optional=True,
        help='Sampling interval at output of response [s].')
    log = List.T(
        Tuple.T(3, String.T()),
        help='Log messages noting problems encountered or simplifications '
             'made while converting response information from its source '
             'format, with entries of the form ``(level, context, '
             'message)``.')

    def __init__(self, **kwargs):
        kwargs['codes'] = CodesNSLCE(kwargs['codes'])
        Object.__init__(self, **kwargs)
        self._effective_responses_cache = {}

    @property
    def time_span(self):
        '''
        Get response's epoch as ``(tmin, tmax)`` tuple.
        '''
        return (self.tmin, self.tmax)

    @property
    def nstages(self):
        '''
        Number of response stages.
        '''
        return len(self.stages)

    @property
    def input_quantity(self):
        '''
        Physical quantity accepted at input of first stage, or ``None`` if
        no stages are present.
        '''
        return self.stages[0].input_quantity if self.stages else None

    @property
    def output_quantity(self):
        '''
        Physical quantity produced at output of last stage, or ``None`` if
        no stages are present.
        '''
        return self.stages[-1].output_quantity if self.stages else None

    @property
    def output_sample_rate(self):
        '''
        Sampling rate at output of last stage [Hz], or ``None`` if no
        stages are present.
        '''
        return self.stages[-1].output_sample_rate if self.stages else None

    @property
    def summary_stages(self):
        '''
        One-line summary of the :py:attr:`stages`' types.
        '''
        def grouped(xs):
            xs = list(xs)
            g = []
            for i in range(len(xs)):
                g.append(xs[i])
                if i+1 < len(xs) and xs[i+1] != xs[i]:
                    yield g
                    g = []

            if g:
                yield g

        return '+'.join(
            '%s%s' % (g[0], '(%i)' % len(g) if len(g) > 1 else '')
            for g in grouped(stage.stage_type for stage in self.stages))

    @property
    def summary_quantities(self):
        '''
        One-line summary of input/output quantities and output sample
        rate.
        '''
        orate = self.output_sample_rate
        return '%s -> %s%s' % (
            self.input_quantity or '?',
            self.output_quantity or '?',
            ' @ %g Hz' % orate if orate else '')

    @property
    def summary_log(self):
        '''
        One-letter-per-entry summary of :py:attr:`log` severities.
        '''
        return ''.join(sorted(set(x[0][0].upper() for x in self.log)))

    @property
    def summary_entries(self):
        '''
        Get tuple of one-line summary values, for use with
        :py:func:`~pyrocko.util.fmt_summary`.
        '''
        return (
            self.__class__.__name__,
            str(self.codes),
            self.str_time_span,
            self.summary_log,
            self.summary_quantities,
            self.summary_stages)

    @property
    def summary(self):
        '''
        One-line summary of the response.
        '''
        return util.fmt_summary(self.summary_entries, (10, 20, 55, 3, 35, 0))

    def get_effective(
            self,
            input_quantity=None,
            stages=(None, None),
            mode='complete',
            gain_frequency=None):

        '''
        Get effective combined frequency response.

        :param input_quantity:
            If given, prepend a conversion from the given physical quantity
            to the response's native :py:attr:`input_quantity`.
        :type input_quantity:
            str

        :param stages:
            Restrict combination to a sub-range of :py:attr:`stages`, given
            as ``(start, stop)`` index tuple, following Python slicing
            conventions. Use ``None`` for an open end.
        :type stages:
            :py:class:`tuple` of 2 :py:class:`int` or ``None``

        :param mode:
            ``'complete'`` to combine all selected stages, or ``'sensor'``
            to instead replace non-sensor stages with a constant gain
            evaluated at ``gain_frequency`` (useful to inspect the sensor's
            response in isolation while still accounting for the overall
            system gain).
        :type mode:
            str

        :param gain_frequency:
            Frequency [Hz] at which to evaluate the constant gain used to
            replace non-sensor stages when ``mode='sensor'``. Required in
            that case.
        :type gain_frequency:
            float

        :returns:
            Combined response.
        :rtype:
            :py:class:`~pyrocko.response.MultiplyResponse`

        Results are cached per ``(input_quantity, stages)`` combination.
        '''

        assert mode in ('complete', 'sensor')
        assert not (mode == 'sensor' and gain_frequency is None)

        cache_key = (input_quantity, stages)
        if cache_key in self._effective_responses_cache:
            return self._effective_responses_cache[cache_key]

        try:
            elements = response_converters(input_quantity, self.input_quantity)
        except ConversionError as e:
            raise ConversionError(str(e) + ' (%s)' % self.summary)

        for istage, stage in enumerate(self.stages):
            if (stages[0] is None or stages[0] <= istage) \
                    and (stages[1] is None or istage < stages[1]):

                is_sensor = istage == 0 or stage.stage_type == 'sensor'

                if mode == 'complete' or (mode == 'sensor' and is_sensor):
                    elements.append(stage.get_effective())
                else:
                    resp = stage.get_effective()
                    gain = Gain(constant=resp.evaluate1(gain_frequency))
                    # maybe check derivative if response is flat enough?
                    # or check frequency band?
                    elements.append(gain)

        if input_quantity is None \
                or input_quantity == self.input_quantity:
            checkpoints = self.checkpoints
        else:
            checkpoints = []

        resp = MultiplyResponse(
            responses=simplify_responses(elements),
            checkpoints=checkpoints)

        self._effective_responses_cache[cache_key] = resp
        return resp


def ehash(s):
    return hashlib.sha1(s.encode('utf8')).hexdigest()


def random_name(n=8):
    return urlsafe_b64encode(urandom(n)).rstrip(b'=').decode('ascii')


@squirrel_content
class WaveformPromise(Object):
    '''
    Information about a waveform potentially downloadable from a remote site.

    In the Squirrel framework, waveform promises are used to permit download of
    selected waveforms from a remote site. They are typically generated by
    calls to
    :py:meth:`~pyrocko.squirrel.base.Squirrel.update_waveform_promises`.
    Waveform promises are inserted and indexed in the database similar to
    normal waveforms. When processing a waveform query, e.g. from
    :py:meth:`~pyrocko.squirrel.base.Squirrel.get_waveforms`, and no local
    waveform is available for the queried time span, a matching promise can be
    resolved, i.e. an attempt is made to download the waveform from the remote
    site. The promise is removed after the download attempt (except when a
    network error occurs). This prevents Squirrel from making unnecessary
    queries for waveforms missing at the remote site.
    '''

    codes = CodesNSLCE.T(
        help='Codes identifying the channel (NET.STA.LOC.CHA.EXTRA) of the '
             'promised waveform.')
    tmin = Timestamp.T(
        help='Start time of promised waveform.')
    tmax = Timestamp.T(
        help='End time of promised waveform.')

    deltat = Float.T(
        optional=True,
        help='Sampling interval of promised waveform [s].')

    source_hash = String.T(
        help='Hash identifying the remote source client providing this '
             'promise.')

    def __init__(self, **kwargs):
        kwargs['codes'] = CodesNSLCE(kwargs['codes'])
        Object.__init__(self, **kwargs)

    @property
    def time_span(self):
        '''
        Get promise's temporal span as ``(tmin, tmax)`` tuple.
        '''
        return (self.tmin, self.tmax)


class InvalidWaveform(Exception):
    '''
    Raised when a downloaded waveform fails validation, see
    :py:meth:`WaveformOrder.validate`.
    '''


class WaveformOrder(Object):
    '''
    Waveform request information.
    '''

    source_id = String.T(
        help='Id of remote source client to query, of the form '
             '``\'client:<client-type>:<hash>\'``, e.g. '
             '``\'client:fdsn:<hash>\'``.')
    codes = CodesNSLCE.T(
        help='Codes identifying the channel (NET.STA.LOC.CHA.EXTRA) to be '
             'downloaded.')
    deltat = Float.T(
        help='Expected sampling interval of the waveform to be downloaded '
             '[s].')
    tmin = Timestamp.T(
        help='Start time of waveform to be downloaded.')
    tmax = Timestamp.T(
        help='End time of waveform to be downloaded.')
    gaps = List.T(
        Tuple.T(2, Timestamp.T()),
        help='List of ``(tmin, tmax)`` time spans already locally '
             'available, to be excluded from the request when possible.')
    time_created = Timestamp.T(
        help='Time at which the order was created.')
    anxious = Duration.T(
        default=600.,
        help='Time span [s] before ``tmax`` within which the order is '
             'still considered to concern near-real-time data, see '
             ':py:meth:`is_near_real_time`.')

    @property
    def client(self):
        '''
        Get remote source client type (e.g. ``'fdsn'``), parsed from
        :py:attr:`source_id`.
        '''
        return self.source_id.split(':')[1]

    def key1(self):
        '''
        Get key identifying the order, including its source, for
        duplicate/repeat-order detection.
        '''
        return (self.source_id, self.codes, self.tmin, self.tmax)

    def key2(self):
        '''
        Get key identifying the order, without its source, e.g. for
        cross-client bookkeeping of already-satisfied requests.
        '''
        return (self.codes, self.tmin, self.tmax)

    def describe(self, site='?'):
        '''
        Get one-line description of the order.

        :param site:
            Name of remote site, for display purposes.
        :type site:
            str

        :rtype:
            str
        '''
        return '%s:%s %s [%s - %s]' % (
            self.client, site, str(self.codes),
            util.time_to_str(self.tmin), util.time_to_str(self.tmax))

    def validate(self, tr):
        '''
        Check that a downloaded trace matches this order.

        :param tr:
            Downloaded waveform to be checked.
        :type tr:
            :py:class:`~pyrocko.trace.Trace`

        :raises InvalidWaveform:
            If the trace has no data samples, a sampling interval not
            matching :py:attr:`deltat`, or non-finite data values.
        '''
        if tr.ydata.size == 0:
            raise InvalidWaveform(
                'waveform with zero data samples')

        if abs(tr.deltat - self.deltat) > (tr.deltat + self.deltat)*1e-5:
            raise InvalidWaveform(
                'incorrect sampling interval - waveform: %15.12e s, '
                'meta-data: %15.12e s' % (
                    tr.deltat, self.deltat))

        if not num.all(num.isfinite(tr.ydata)):
            raise InvalidWaveform('waveform has NaN values')

    def estimate_nsamples(self):
        '''
        Estimate number of samples expected for the ordered time span.
        '''
        return int(round((self.tmax - self.tmin) / self.deltat))+1

    def is_near_real_time(self):
        '''
        Check if order's end time is close enough to its creation time to
        be considered a near-real-time request (see :py:attr:`anxious`).
        '''
        return self.tmax > self.time_created - self.anxious


def order_summary(orders):
    codes_list = sorted(set(order.codes for order in orders))
    if len(codes_list) > 3:
        return '%i order%s: %s - %s' % (
            len(orders),
            util.plural_s(orders),
            str(codes_list[0]),
            str(codes_list[1]))

    else:
        return '%i order%s: %s' % (
            len(orders),
            util.plural_s(orders),
            ', '.join(str(codes) for codes in codes_list))


class Nut(Object):
    '''
    Index entry referencing an elementary piece of content.

    So-called *nuts* are used in Pyrocko's Squirrel framework to hold common
    meta-information about individual pieces of waveforms, stations, channels,
    etc. together with the information where it was found or generated.
    '''

    file_path = String.T(
        optional=True,
        help='Path of file containing the referenced content.')
    file_format = String.T(
        optional=True,
        help='Content type of file, as used by the Squirrel IO backends '
             '(e.g. ``\'mseed\'``, ``\'stationxml\'``, ``\'yaml\'``).')
    file_mtime = Timestamp.T(
        optional=True,
        help='Modification time of file, for change detection.')
    file_size = Int.T(
        optional=True,
        help='Size of file [bytes], for change detection.')

    file_segment = Int.T(
        optional=True,
        help='Index of segment within file, for file formats where a '
             'single file can contain multiple segments.')
    file_element = Int.T(
        optional=True,
        help='Index of content element within its segment, for content '
             'types where a segment may hold more than one element.')

    kind_id = Int.T(
        help='Content type, one of the ids given in '
             ':py:data:`g_content_kind_ids`.')
    codes = Codes.T(
        help='Codes identifying the content, of a subclass appropriate for '
             'the given content kind.')

    tmin_seconds = Int.T(
        default=0,
        help='Full seconds part of start time of content, needed for fast '
             'database queries.')
    tmin_offset = Int.T(
        default=0,
        optional=True,
        help='Fractional/high-precision part of start time of content '
             '(see :py:func:`tsplit`).')
    tmax_seconds = Int.T(
        default=0,
        help='Full seconds part of end time of content, needed for fast '
             'database queries.')
    tmax_offset = Int.T(
        default=0,
        optional=True,
        help='Fractional/high-precision part of end time of content '
             '(see :py:func:`tsplit`).')

    deltat = Float.T(
        default=0.0,
        help='Sampling interval of content, if applicable, else ``0.0`` '
             '[s].')

    content = Any.T(
        optional=True,
        help='The actual content object represented by this nut, if '
             'attached.')
    raw_content = Dict.T(
        String.T(), Any.T(),
        help='Additional, backend-specific raw content, keyed by content '
             'kind.')

    content_in_db = False

    def __init__(
            self,
            file_path=None,
            file_format=None,
            file_mtime=None,
            file_size=None,
            file_segment=None,
            file_element=None,
            kind_id=0,
            codes=CodesX(''),
            tmin_seconds=None,
            tmin_offset=0,
            tmax_seconds=None,
            tmax_offset=0,
            deltat=None,
            content=None,
            raw_content=None,
            tmin=None,
            tmax=None,
            values_nocheck=None):

        '''
        Create a nut.

        :param file_path:
            Path of file containing the referenced content.
        :type file_path:
            str

        :param file_format:
            Content type of file, as used by the Squirrel IO backends.
        :type file_format:
            str

        :param file_mtime:
            Modification time of file, for change detection.
        :type file_mtime:
            :py:func:`~pyrocko.util.get_time_float`

        :param file_size:
            Size of file [bytes], for change detection.
        :type file_size:
            int

        :param file_segment:
            Index of segment within file.
        :type file_segment:
            int

        :param file_element:
            Index of content element within its segment.
        :type file_element:
            int

        :param kind_id:
            Content type, one of the ids given in
            :py:data:`g_content_kind_ids`.
        :type kind_id:
            int

        :param codes:
            Codes identifying the content, of a class appropriate for
            ``kind_id`` (converted with :py:func:`to_codes` if needed).

        :param tmin:
            Start time of content. Alternative to giving ``tmin_seconds``/
            ``tmin_offset`` directly.
        :type tmin:
            :py:func:`~pyrocko.util.get_time_float`

        :param tmax:
            End time of content. Alternative to giving ``tmax_seconds``/
            ``tmax_offset`` directly.
        :type tmax:
            :py:func:`~pyrocko.util.get_time_float`

        :param deltat:
            Sampling interval of content, if applicable.
        :type deltat:
            float

        :param content:
            The actual content object represented by this nut, if to be
            attached directly.

        :param raw_content:
            Additional, backend-specific raw content, keyed by content
            kind.
        :type raw_content:
            dict

        :param values_nocheck:
            If given, a tuple of already-validated field values (as
            produced by a database row), used to bypass normal argument
            processing for fast, low-level construction (internal use).
        :type values_nocheck:
            tuple
        '''

        if values_nocheck is not None:
            (self.file_path, self.file_format, self.file_mtime,
             self.file_size,
             self.file_segment, self.file_element,
             self.kind_id, codes_safe_str,
             self.tmin_seconds, self.tmin_offset,
             self.tmax_seconds, self.tmax_offset,
             self.deltat) = values_nocheck

            self.codes = to_codes_simple(self.kind_id, codes_safe_str)
            self.deltat = self.deltat or None
            self.raw_content = {}
            self.content = None
        else:
            if tmin is not None:
                tmin_seconds, tmin_offset = tsplit(tmin)

            if tmax is not None:
                tmax_seconds, tmax_offset = tsplit(tmax)

            self.kind_id = int(kind_id)
            self.codes = codes
            self.tmin_seconds = int_or_g_tmin(tmin_seconds)
            self.tmin_offset = int(tmin_offset)
            self.tmax_seconds = int_or_g_tmax(tmax_seconds)
            self.tmax_offset = int(tmax_offset)
            self.deltat = float_or_none(deltat)
            self.file_path = str_or_none(file_path)
            self.file_segment = int_or_none(file_segment)
            self.file_element = int_or_none(file_element)
            self.file_format = str_or_none(file_format)
            self.file_mtime = float_or_none(file_mtime)
            self.file_size = int_or_none(file_size)
            self.content = content
            if raw_content is None:
                self.raw_content = {}
            else:
                self.raw_content = raw_content

        Object.__init__(self, init_props=False)

    def __eq__(self, other):
        return (isinstance(other, Nut) and
                self.equality_values == other.equality_values)

    def hash(self):
        '''
        Compute hash from :py:attr:`key`, identifying this nut's file
        origin.

        :rtype:
            str
        '''
        return ehash(','.join(str(x) for x in self.key))

    def __ne__(self, other):
        return not (self == other)

    def get_io_backend(self):
        '''
        Get the Squirrel IO backend module responsible for
        :py:attr:`file_format`.
        '''
        from . import io
        return io.get_backend(self.file_format)

    def file_modified(self):
        '''
        Check if the referenced file appears to have changed since
        indexing, by comparing current vs. indexed modification time and
        size.

        :rtype:
            bool
        '''
        return self.get_io_backend().get_stats(self.file_path) \
            != (self.file_mtime, self.file_size)

    @property
    def dkey(self):
        '''
        Get key used to index this nut in the database's content table,
        as ``(kind_id, tmin_seconds, tmin_offset, codes)``.
        '''
        return (self.kind_id, self.tmin_seconds, self.tmin_offset, self.codes)

    @property
    def key(self):
        '''
        Get key identifying this nut's file origin, as
        ``(file_path, file_segment, file_element, file_mtime)``.
        '''
        return (
            self.file_path,
            self.file_segment,
            self.file_element,
            self.file_mtime)

    @property
    def equality_values(self):
        '''
        Get values determining content equality (used by ``__eq__``), as
        ``(file_segment, file_element, kind_id, codes, tmin_seconds,
        tmin_offset, tmax_seconds, tmax_offset, deltat)``.
        '''
        return (
            self.file_segment, self.file_element,
            self.kind_id, self.codes,
            self.tmin_seconds, self.tmin_offset,
            self.tmax_seconds, self.tmax_offset, self.deltat)

    def diff(self, other):
        '''
        Get list of differing :py:attr:`equality_values` fields between
        this and another nut.

        :param other:
            Nut to compare against.
        :type other:
            :py:class:`Nut`

        :returns:
            ``(field_name, this_value, other_value)`` tuples for each
            differing field.
        :rtype:
            :py:class:`list` of :py:class:`tuple`
        '''
        names = [
            'file_segment', 'file_element', 'kind_id', 'codes',
            'tmin_seconds', 'tmin_offset', 'tmax_seconds', 'tmax_offset',
            'deltat']

        d = []
        for name, a, b in zip(
                names, self.equality_values, other.equality_values):

            if a != b:
                d.append((name, a, b))

        return d

    @property
    def tmin(self):
        '''
        Start time of content, joined from :py:attr:`tmin_seconds`/
        :py:attr:`tmin_offset` (see :py:func:`tjoin`). Settable, splitting
        into those two fields (see :py:func:`tsplit`).
        '''
        return tjoin(self.tmin_seconds, self.tmin_offset)

    @tmin.setter
    def tmin(self, t):
        self.tmin_seconds, self.tmin_offset = tsplit(t)

    @property
    def tmax(self):
        '''
        End time of content, joined from :py:attr:`tmax_seconds`/
        :py:attr:`tmax_offset` (see :py:func:`tjoin`). Settable, splitting
        into those two fields (see :py:func:`tsplit`).
        '''
        return tjoin(self.tmax_seconds, self.tmax_offset)

    @tmax.setter
    def tmax(self, t):
        self.tmax_seconds, self.tmax_offset = tsplit(t)

    @property
    def kscale(self):
        '''
        Time-block granularity index for this nut's time span, used for
        indexing content at multiple scales (see :py:data:`tscale_edges`
        and :py:meth:`~pyrocko.squirrel.base.Squirrel.iter_nuts`'s
        ``tscale_min`` parameter).
        '''
        if self.tmin_seconds is None or self.tmax_seconds is None:
            return 0
        return tscale_to_kscale(self.tmax_seconds - self.tmin_seconds)

    @property
    def waveform_kwargs(self):
        '''
        Get keyword arguments usable to create a
        :py:class:`~pyrocko.trace.Trace` from this nut's meta-information.
        '''
        network, station, location, channel, extra = self.codes

        return dict(
            network=network,
            station=station,
            location=location,
            channel=channel,
            extra=extra,
            tmin=self.tmin,
            tmax=self.tmax,
            deltat=self.deltat)

    @property
    def waveform_promise_kwargs(self):
        '''
        Get keyword arguments usable to create a
        :py:class:`WaveformPromise` from this nut's meta-information.
        '''
        return dict(
            codes=self.codes,
            tmin=self.tmin,
            tmax=self.tmax,
            deltat=self.deltat)

    @property
    def station_kwargs(self):
        '''
        Get keyword arguments usable to create a :py:class:`Station` from
        this nut's meta-information.
        '''
        network, station, location = self.codes
        return dict(
            codes=self.codes,
            tmin=tmin_or_none(self.tmin),
            tmax=tmax_or_none(self.tmax))

    @property
    def channel_kwargs(self):
        '''
        Get keyword arguments usable to create a :py:class:`Channel` from
        this nut's meta-information.
        '''
        network, station, location, channel, extra = self.codes
        return dict(
            codes=self.codes,
            tmin=tmin_or_none(self.tmin),
            tmax=tmax_or_none(self.tmax),
            deltat=self.deltat)

    @property
    def response_kwargs(self):
        '''
        Get keyword arguments usable to create a :py:class:`Response` from
        this nut's meta-information.
        '''
        return dict(
            codes=self.codes,
            tmin=tmin_or_none(self.tmin),
            tmax=tmax_or_none(self.tmax),
            deltat=self.deltat)

    @property
    def event_kwargs(self):
        '''
        Get keyword arguments usable to create a
        :py:class:`~pyrocko.model.event.Event` from this nut's
        meta-information.
        '''
        return dict(
            name=self.codes,
            time=self.tmin,
            duration=(self.tmax - self.tmin) or None)

    @property
    def trace_kwargs(self):
        '''
        Get keyword arguments usable to create a
        :py:class:`~pyrocko.trace.Trace` matching this nut's meta-
        information.

        :py:class:`~pyrocko.trace.Trace`'s ``tmax`` is the time of the last
        sample while :py:class:`Nut`'s ``tmax`` is one delta more to simplify
        continuity checks.
        '''
        network, station, location, channel, extra = self.codes

        return dict(
            network=network,
            station=station,
            location=location,
            channel=channel,
            extra=extra,
            tmin=self.tmin,
            tmax=self.tmax-self.deltat,
            deltat=self.deltat)

    @property
    def dummy_trace(self):
        '''
        Get a lightweight, meta-information-only stand-in for a
        :py:class:`~pyrocko.trace.Trace`.

        :rtype:
            :py:class:`DummyTrace`
        '''
        return DummyTrace(self)

    @property
    def summary_entries(self):
        '''
        Get tuple of one-line summary values, for use with
        :py:func:`~pyrocko.util.fmt_summary`.
        '''
        if self.tmin == self.tmax:
            ts = util.time_to_str(self.tmin)
        else:
            ts = '%s - %s' % (
                util.time_to_str(self.tmin),
                util.time_to_str(self.tmax))

        return (
            self.__class__.__name__,
            to_kind(self.kind_id),
            str(self.codes),
            ts)

    @property
    def summary(self):
        '''
        One-line summary of the nut.
        '''
        return util.fmt_summary(self.summary_entries, (10, 16, 20, 0))


def make_waveform_nut(**kwargs):
    return Nut(kind_id=WAVEFORM, **kwargs)


def make_waveform_promise_nut(**kwargs):
    return Nut(kind_id=WAVEFORM_PROMISE, **kwargs)


def make_station_nut(**kwargs):
    return Nut(kind_id=STATION, **kwargs)


def make_channel_nut(**kwargs):
    return Nut(kind_id=CHANNEL, **kwargs)


def make_response_nut(**kwargs):
    return Nut(kind_id=RESPONSE, **kwargs)


def make_event_nut(**kwargs):
    return Nut(kind_id=EVENT, **kwargs)


def make_carpet_nut(**kwargs):
    return Nut(kind_id=CARPET, **kwargs)


def make_elk_nut(**kwargs):
    return Nut(kind_id=ELK, **kwargs)


def group_channels(nuts):
    by_ansl = {}
    for nut in nuts:
        if nut.kind_id != CHANNEL:
            continue

        ansl = nut.codes[:4]

        if ansl not in by_ansl:
            by_ansl[ansl] = {}

        group = by_ansl[ansl]

        k = nut.codes[4][:-1], nut.deltat, nut.tmin, nut.tmax

        if k not in group:
            group[k] = set()

        group.add(nut.codes[4])

    return by_ansl


class DummyTrace(object):
    '''
    Lightweight, meta-information-only stand-in for a
    :py:class:`~pyrocko.trace.Trace`, backed by a :py:class:`Nut`.

    Exposes the subset of :py:class:`~pyrocko.trace.Trace`'s interface
    (``tmin``, ``tmax``, ``deltat``, ``network``/``station``/``location``/
    ``channel``/``extra``, ``nslc_id``, :py:meth:`overlaps`) needed where a
    trace-like object is expected but no waveform data has been loaded,
    see :py:attr:`Nut.dummy_trace`.
    '''

    def __init__(self, nut):
        '''
        :param nut:
            Nut to represent.
        :type nut:
            :py:class:`Nut`
        '''
        self.nut = nut
        self.codes = nut.codes
        self.meta = {}

    @property
    def tmin(self):
        return self.nut.tmin

    @property
    def tmax(self):
        return self.nut.tmax

    @property
    def deltat(self):
        return self.nut.deltat or 0.0

    @property
    def nslc_id(self):
        return self.codes.nslc

    @property
    def network(self):
        return self.codes.network

    @property
    def station(self):
        return self.codes.station

    @property
    def location(self):
        return self.codes.location

    @property
    def channel(self):
        return self.codes.channel

    @property
    def extra(self):
        return self.codes.extra

    def overlaps(self, tmin, tmax):
        '''
        Check if the represented nut's time span overlaps with a given
        time span.

        :param tmin:
            Start time of query interval.
        :type tmin:
            :py:func:`~pyrocko.util.get_time_float`

        :param tmax:
            End time of query interval.
        :type tmax:
            :py:func:`~pyrocko.util.get_time_float`

        :rtype:
            bool
        '''
        return not (tmax < self.nut.tmin or self.nut.tmax < tmin)


def duration_to_str(t):
    if t > 24*3600:
        return '%gd' % (t / (24.*3600.))
    elif t > 3600:
        return '%gh' % (t / 3600.)
    elif t > 60:
        return '%gm' % (t / 60.)
    else:
        return '%gs' % t


class Coverage(Object):
    '''
    Information about times covered by a waveform or other time series data.
    '''
    kind_id = Int.T(
        help='Content type.',
        optional=True)
    pattern = Codes.T(
        help='The codes pattern in the request, which caused this entry to '
             'match.',
        optional=True)
    codes = Codes.T(
        help='NSLCE or NSL codes identifier of the time series.',
        optional=True)
    deltat = Float.T(
        help='Sampling interval [s]',
        optional=True)
    tmin = Timestamp.T(
        help='Global start time of time series.',
        optional=True)
    tmax = Timestamp.T(
        help='Global end time of time series.',
        optional=True)
    changes = List.T(
        Tuple.T(2, Any.T()),
        help='List of change points, with entries of the form '
             '``(time, count)``, where a ``count`` of zero indicates start of '
             'a gap, a value of 1 start of normal data coverage and a higher '
             'value duplicate or redundant data coverage.',
        optional=True)

    @classmethod
    def from_values(cls, args):
        '''
        Create coverage object from a
        ``(pattern, codes, deltat, tmin, tmax, changes, kind_id)`` tuple
        (internal use).
        '''
        pattern, codes, deltat, tmin, tmax, changes, kind_id = args
        return cls(
            kind_id=kind_id,
            pattern=pattern,
            codes=codes,
            deltat=deltat,
            tmin=tmin,
            tmax=tmax,
            changes=changes)

    @property
    def summary_entries(self):
        '''
        Get tuple of one-line summary values, for use with
        :py:func:`~pyrocko.util.fmt_summary`.
        '''
        ts = '%s - %s' % (
            util.time_to_str(self.tmin),
            util.time_to_str(self.tmax))

        srate = self.sample_rate

        total = self.total

        return (
            self.__class__.__name__,
            to_kind(self.kind_id) if self.kind_id is not None else 'none',
            str(self.codes) if self.codes is not None else 'none',
            ts,
            '%10.3g' % srate if srate else '',
            '%i' % len(self.changes) if self.changes is not None else 'none',
            '%s' % duration_to_str(total) if total else 'none')

    @property
    def summary(self):
        '''
        One-line summary of the coverage.
        '''
        return util.fmt_summary(
            self.summary_entries,
            (10, 16, 20, 55, 10, 4, 0))

    @property
    def sample_rate(self):
        '''
        Sampling rate [Hz], derived from :py:attr:`deltat`, or ``None`` if
        unset.
        '''
        if self.deltat is None:
            return None
        elif self.deltat == 0.0:
            return 0.0
        else:
            return 1.0 / self.deltat

    @property
    def labels(self):
        '''
        Get ``(codes, sample_rate)`` tuple of display labels, e.g. for use
        as plot annotations.
        '''
        srate = self.sample_rate
        return (
            ('%s' % str(self.codes)),
            '%.4g' % srate if srate else '')

    @property
    def total(self):
        '''
        Total duration [s] covered by normal (non-gap) data, or ``None``
        if :py:attr:`changes` is unset.
        '''
        total_t = None
        for tmin, tmax, _ in self.iter_spans():
            total_t = (total_t or 0.0) + (tmax - tmin)

        return total_t

    def iter_spans(self):
        '''
        Iterate over covered time spans.

        :yields:
            ``(tmin, tmax, count)`` tuples, where ``count`` is the number
            of overlapping/redundant data pieces covering the span (1 for
            normal coverage, higher for duplicate/redundant data).
        '''
        if self.changes is None:
            return

        last = None
        for (t, count) in self.changes:
            if last is not None:
                last_t, last_count = last
                if last_count > 0:
                    yield last_t, t, last_count

            last = (t, count)

    def iter_uncovered_by(self, other):
        '''
        Iterate over time spans covered here but not in another coverage.

        :param other:
            Coverage to compare against.
        :type other:
            :py:class:`Coverage`

        :yields:
            ``(tmin, tmax, ia, ib)`` tuples, where ``ia``/``ib`` are the
            indices into this/the other coverage's :py:attr:`changes` at
            the start of the yielded span.
        '''
        if None in (self.changes, other.changes):
            return

        a = self
        b = other
        ia = ib = -1
        ca = cb = 0
        last = None
        while not (ib + 1 == len(b.changes) and ia + 1 == len(a.changes)):
            if ib + 1 == len(b.changes):
                ia += 1
                t, ca = a.changes[ia]
            elif ia + 1 == len(a.changes):
                ib += 1
                t, cb = b.changes[ib]
            elif a.changes[ia+1][0] < b.changes[ib+1][0]:
                ia += 1
                t, ca = a.changes[ia]
            else:
                ib += 1
                t, cb = b.changes[ib]

            if last is not None:
                tl, cal, cbl = last
                if tl < t and cal > 0 and cbl == 0:
                    yield tl, t, ia, ib

            last = (t, ca, cb)

    def iter_uncovered_by_combined(self, other):
        '''
        Like :py:meth:`iter_uncovered_by`, but merging adjacent spans
        belonging to the same gap in ``other``.

        :param other:
            Coverage to compare against.
        :type other:
            :py:class:`Coverage`

        :yields:
            ``(tmin, tmax)`` tuples.
        '''
        ib_last = None
        group = None
        for tmin, tmax, _, ib in self.iter_uncovered_by(other):
            if ib_last is None or ib != ib_last:
                if group:
                    yield (group[0][0], group[-1][1])

                group = []

            group.append((tmin, tmax))
            ib_last = ib

        if group:
            yield (group[0][0], group[-1][1])

    def changes_as_arrays(self):
        '''
        Get :py:attr:`changes` as a pair of NumPy arrays.

        :returns:
            ``(times, counts)``, or ``(None, None)`` if :py:attr:`changes`
            is unset.
        :rtype:
            :py:class:`tuple` of 2 :py:class:`numpy.ndarray`
        '''
        if self.changes is None:
            return None, None

        time_float = util.get_time_float()
        times = num.array([t for (t, _) in self.changes], dtype=time_float)
        counts = num.array([c for (_, c) in self.changes], dtype=int)
        return times, counts

    def contiguous(self, tmin, tmax):
        '''
        Get coverage count for a time span, if constant throughout.

        :param tmin:
            Start of time span to check.
        :type tmin:
            :py:func:`~pyrocko.util.get_time_float`

        :param tmax:
            End of time span to check.
        :type tmax:
            :py:func:`~pyrocko.util.get_time_float`

        :returns:
            The coverage count if it is constant across ``[tmin, tmax]``,
            or ``-1`` if it changes within the span or is unknown.
        :rtype:
            int
        '''

        if self.changes is None or len(self.changes) < 1:
            return -1

        count = -1
        for t, c in self.changes:
            if t <= tmin:
                count = c
            elif tmin < t and t < tmax:
                if c != count:
                    return -1
            else:
                return count

        return count

    def get(self, time):
        '''
        Get coverage count at a given time instant.

        :param time:
            Time instant to query.
        :type time:
            :py:func:`~pyrocko.util.get_time_float`

        :rtype:
            int
        '''
        for (t, v) in self.changes:
            if t <= time:
                return v

        return 0


def same_or_none(xs):
    xs = list(xs)
    if not xs:
        return None

    if all(x == xs[0] for x in xs):
        return xs[0]
    else:
        return None


def join_coverages(coverages, tbleed=0.0):
    '''
    Combine several coverages into one, merging their change points.

    :param coverages:
        Coverages to be combined, e.g. of several channels belonging to
        the same sensor, for which overall combined data availability is
        of interest.
    :type coverages:
        iterable of :py:class:`Coverage` objects

    :param tbleed:
        Time interval to be removed at coverage edges and gaps [s]. This can be
        used, e.g. to track edge effect propagation through processing steps.
    :type tbleed:
        float

    :returns:
        Combined coverage. Its :py:attr:`~Coverage.kind_id`/
        :py:attr:`~Coverage.deltat` are set only if all input coverages
        agree on them, else left unset (``None``); its
        :py:attr:`~Coverage.pattern`/:py:attr:`~Coverage.codes` are always
        left unset.
    :rtype:
        :py:class:`Coverage`

    :raises pyrocko.trace.NoData:
        If ``coverages`` is empty, or the combined time span is empty.
    '''
    from pyrocko import trace

    if not coverages:
        raise trace.NoData

    tmin = min(coverage.tmin for coverage in coverages) + tbleed
    tmax = max(coverage.tmax for coverage in coverages) - tbleed

    if tmax < tmin:
        raise trace.NoData

    kind_id = same_or_none(coverage.kind_id for coverage in coverages)
    deltat = same_or_none(coverage.deltat for coverage in coverages)

    if any(coverage.changes is None for coverage in coverages):
        changes = None
    else:
        all_times = []
        all_diff_counts = []
        for coverage in coverages:
            times, counts = coverage.changes_as_arrays()
            diff_counts = counts.copy()
            diff_counts[0] = counts[0]
            diff_counts[1:] = counts[1:] - counts[:-1]
            all_diff_counts.append(diff_counts)
            if tbleed != 0.0:
                times[diff_counts > 0] += tbleed
                times[diff_counts < 0] -= tbleed

            all_times.append(times)

        times = num.concatenate(all_times)
        diff_counts = num.concatenate(all_diff_counts)
        iorder = num.argsort(times)
        times = times[iorder]
        diff_counts = diff_counts[iorder]
        counts = num.cumsum(diff_counts)
        changes = list(zip(times, counts))

    coverage = Coverage(
        kind_id=kind_id,
        tmin=tmin,
        tmax=tmax,
        deltat=deltat,
        changes=changes)

    return coverage


def make_rich_coverage(kinds, coverages):
    '''
    Combine several coverages of the same codes but different content
    kinds into one, encoding each kind's coverage state in separate bits
    of the combined :py:attr:`~Coverage.changes` counts.

    :param kinds:
        Content kinds to encode, in the order their coverage states are
        packed into the combined counts (2 bits per kind, low bit first,
        see :py:meth:`~pyrocko.squirrel.base.Squirrel.get_rich_coverage`
        for the kinds and their order used there).
    :type kinds:
        :py:class:`list` of :py:class:`str`

    :param coverages:
        Coverages to be combined, all sharing the same
        :py:attr:`~Coverage.codes`, of the content kinds given in
        ``kinds`` (any kind may be represented multiple times, or not at
        all).
    :type coverages:
        iterable of :py:class:`Coverage` objects

    :returns:
        Combined coverage with ``kind_id=None`` and each change point's
        count encoding, per kind, ``0`` for no coverage, ``1`` for normal
        coverage, ``2`` for duplicate/redundant coverage or ``3`` if the
        coverage state could not be determined (see
        :py:attr:`~Coverage.changes`).
    :rtype:
        :py:class:`Coverage`
    '''
    assert len(coverages) >= 1
    assert all(coverage.codes == coverages[0].codes for coverage in coverages)
    tmin = min(coverage.tmin for coverage in coverages)
    tmax = max(coverage.tmax for coverage in coverages)

    kind_ids = to_kind_ids(kinds)

    all_times = []
    all_diff_counts = []
    all_ikinds = []
    coverages_undefined = []

    for coverage in coverages:
        if coverage.changes is None:
            coverages_undefined.append(coverage)
            times = num.array([coverage.tmin, coverage.tmax])
            diff_counts = num.array([1, -1])
        else:
            times, counts = coverage.changes_as_arrays()
            diff_counts = counts.copy()
            diff_counts[0] = counts[0]
            diff_counts[1:] = counts[1:] - counts[:-1]

        ikinds = num.full(
            times.size,
            kind_ids.index(coverage.kind_id),
            dtype=int)

        all_times.append(times)
        all_diff_counts.append(diff_counts)
        all_ikinds.append(ikinds)

    times = num.concatenate(all_times)
    diff_counts = num.concatenate(all_diff_counts)
    ikinds = num.concatenate(all_ikinds)

    iorder = num.argsort(times)

    times = times[iorder]
    diff_counts = diff_counts[iorder]
    ikinds = ikinds[iorder]

    diff_counts2 = num.zeros((diff_counts.size, len(kind_ids)), dtype=int)
    diff_counts2[num.arange(times.size), ikinds] = diff_counts

    counts2 = num.cumsum(diff_counts2, axis=0)
    counts2 = num.clip(counts2, 0, 2)

    for coverage in coverages_undefined:
        mask = num.logical_and(coverage.tmin <= times, times < coverage.tmax)
        counts2[mask, kind_ids.index(coverage.kind_id)] = 3

    counts = num.zeros(times.size, dtype=int)
    for ikind, kind_id in enumerate(kind_ids):
        counts += counts2[:, ikind] << (2 * ikind)

    tmask = num.ones(times.size, dtype=bool)
    tmask[0:-1] = num.diff(times) != 0.0

    times = times[tmask]
    counts = counts[tmask]

    changes = list(zip(times, counts))

    deltat = same_or_none(
        coverage.deltat for coverage in coverages
        if coverage.deltat is not None)

    coverage = Coverage(
        kind_id=None,
        codes=coverages[0].codes,
        tmin=tmin,
        tmax=tmax,
        deltat=deltat,
        changes=changes)

    return coverage


def unpack_rich(value):
    return [(value >> (2 * i)) & 3 for i in range(16)]


def pack_rich(values):
    return sum(v << (2 * i) for (i, v) in enumerate(values))


__all__ = [
    'UNDEFINED',
    'WAVEFORM',
    'STATION',
    'CHANNEL',
    'RESPONSE',
    'EVENT',
    'WAVEFORM_PROMISE',
    'EMPTY',
    'CARPET',
    'ELK',
    'to_codes',
    'to_codes_guess',
    'to_codes_simple',
    'to_kind',
    'to_kinds',
    'to_kind_id',
    'to_kind_ids',
    'merge_codes',
    'match_codes',
    'codes_patterns_for_kind',
    'CodesMatcher',
    'CodesError',
    'Codes',
    'CodesNSLCE',
    'CodesNSL',
    'CodesX',
    'Station',
    'Channel',
    'Sensor',
    'Response',
    'Nut',
    'Coverage',
    'join_coverages',
    'make_rich_coverage',
    'WaveformPromise',
    'QuantityType',
]
