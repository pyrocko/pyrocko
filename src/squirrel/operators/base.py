# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import annotations

from typing import TYPE_CHECKING, Generator
if TYPE_CHECKING:
    from ..base import Squirrel

from collections import defaultdict
import logging
import re

from pyrocko.model import Location

from ..model import QuantityType, CodesNSLCE, CodesMatcher, CHANNEL, \
    get_selection_args, Sensor

from pyrocko.guts import Object, String, Duration, Float, clone, List

guts_prefix = 'squirrel.ops'

logger = logging.getLogger('psq.ops')


def odiff(a, b):
    ia = ib = 0
    only_a = []
    only_b = []
    while ia < len(a) or ib < len(b):
        # TODO remove when finished with implementation
        if ia > 0:
            assert a[ia] > a[ia-1]
        if ib > 0:
            assert b[ib] > b[ib-1]

        if ib == len(b) or (ia < len(a) and a[ia] < b[ib]):
            only_a.append(a[ia])
            ia += 1
        elif ia == len(a) or (ib < len(b) and a[ia] > b[ib]):
            only_b.append(b[ib])
            ib += 1
        elif a[ia] == b[ib]:
            ia += 1
            ib += 1

    return only_a, only_b


def _cglob_translate(creg):
    dd = []
    for c in creg:
        if c == '*':
            d = r'[^.]*'
        elif c == '?':
            d = r'[^.]'
        elif c == '.':
            d = r'\.'
        else:
            d = c

        dd.append(d)
    reg = ''.join(dd)
    return reg


def scodes(codes):
    css = list(zip(*codes))
    if sum(not all(c == cs[0] for c in cs) for cs in css) == 1:
        return '.'.join(
            cs[0] if all(c == cs[0] for c in cs) else '(%s)' % ','.join(cs)
            for cs in css)
    else:
        return ', '.join(str(c) for c in codes)


class Filtering(Object):
    '''
    Base class for :py:class:`pyrocko.squirrel.model.Nut` filters.
    '''

    def filter(self, it):
        return list(it)


class RegexFiltering(Filtering):
    '''
    Filter by regex.
    '''
    pattern = String.T(default=r'(.*)')

    def __init__(self, **kwargs):
        Filtering.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def filter(self, it):
        return list(filter(self._compiled_pattern.fullmatch), it)


class CodesPatternFiltering(Filtering):
    '''
    Filter by codes pattern.
    '''
    codes = List.T(CodesNSLCE.T(), optional=True)

    def __init__(self, **kwargs):
        Filtering.__init__(self, **kwargs)
        if self.codes is not None:
            self._matcher = CodesMatcher(self.codes)
        else:
            self._matcher = None

    def match(self, codes):
        return True if self._matcher is None else self._matcher.match(codes)

    def filter(self, it):
        if self._matcher is None:
            return list(it)
        else:
            return list(self._matcher.filter(it))


class Grouping(Object):
    '''
    Base class for :py:class:`pyrocko.squirrel.model.Nut` grouping mechanisms.
    '''

    def key(self, codes):
        return codes


class RegexGrouping(Grouping):
    '''
    Group by regex pattern.
    '''
    pattern = String.T(default=r'(.*)')

    def __init__(self, **kwargs):
        Grouping.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def key(self, codes):
        return self._compiled_pattern.fullmatch(codes.safe_str).groups()


class NetworkGrouping(RegexGrouping):
    '''
    Group by *network* code.
    '''
    pattern = String.T(default=_cglob_translate('(*).*.*.*.*'))


class StationGrouping(RegexGrouping):
    '''
    Group by *network.station* codes.
    '''
    pattern = String.T(default=_cglob_translate('(*.*).*.*.*'))


class LocationGrouping(RegexGrouping):
    '''
    Group by *network.station.location* codes.
    '''
    pattern = String.T(default=_cglob_translate('(*.*.*).*.*'))


class ChannelGrouping(RegexGrouping):
    '''
    Group by *network.station.location.channel* codes.

    This effectively groups all processings of a channel, which may differ in
    the *extra* codes attribute.
    '''
    pattern = String.T(default=_cglob_translate('(*.*.*.*).*'))


class SensorGrouping(RegexGrouping):
    '''
    Group by *network.station.location.sensor* and *extra* codes.

    For *sensor* all but the last character of the channel code (indicating the
    component) are used. This effectively groups all components of a sensor,
    or processings of a sensor.
    '''
    pattern = String.T(default=_cglob_translate('(*.*.*.*)?(.*)'))


class Translation(Object):
    '''
    Base class for :py:class:`pyrocko.squirrel.model.Nut` translators.
    '''

    def translate(self, codes):
        return codes


class AddSuffixTranslation(Translation):
    '''
    Add a suffix to :py:attr:`~pyrocko.squirrel.model.CodesNSLCEBase.extra`.
    '''
    suffix = String.T(default='')

    def translate(self, codes):
        return codes.replace(extra=codes.extra + self.suffix)


class RegexTranslation(AddSuffixTranslation):
    '''
    Translate :py:class:`pyrocko.squirrel.model.Codes` using a regular
    expression.
    '''
    pattern = String.T(default=r'(.*)')
    replacement = String.T(default=r'\1')

    def __init__(self, **kwargs):
        AddSuffixTranslation.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def translate(self, codes):
        return AddSuffixTranslation.translate(
            self,
            codes.__class__(
                self._compiled_pattern.sub(self.replacement, codes.safe_str)))


class ReplaceComponentTranslation(RegexTranslation):
    '''
    Translate :py:class:`pyrocko.squirrel.model.Codes` by replacing a
    component.
    '''
    pattern = String.T(default=_cglob_translate('(*.*.*.*)?(.*)'))
    replacement = String.T(default=r'\1{component}\2')


class CodesMapping:
    __slots__ = ['in_codes_set', 'in_codes', 'out_codes']

    def __init__(self):
        self.in_codes_set = set()
        self.in_codes = ()
        self.out_codes = ()


class Operator(Object):

    filtering = Filtering.T(default=Filtering.D())
    grouping = Grouping.T(default=Grouping.D())
    translation = Translation.T(default=Translation.D())

    kind_provides = ('channel', 'response', 'waveform')
    kind_requires = ()

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._mappings = {}
        self._available = []
        self._input = None

    @property
    def name(self):
        return self.__class__.__name__

    def set_input(self, input: Operator | Squirrel) -> None:
        self._input = input

    def set_parameters(self, parameters: Parameters) -> None:
        self._parameters = parameters

    def describe(self) -> str:
        return '%s\n  provides: %s\n  requires: %s\n%s' % (
            self.name,
            ', '.join(self.kind_provides),
            ', '.join(self.kind_requires),
            self._str_mappings)

    @property
    def _str_mappings(self) -> str:
        return '\n'.join([
            '  %s <- %s' % (
                scodes(mapping.out_codes),
                scodes(mapping.in_codes))
            for mapping in self.iter_mappings()])

    def translate_codes(self, in_codes: list[CodesNSLCE]) -> list[CodesNSLCE]:
        return [self.translation.translate(codes) for codes in in_codes]

    def iter_mappings(
                self,
                codes: list[CodesNSLCE] | None = None
            ) -> Generator[CodesMapping]:
        if codes:
            cpf = CodesPatternFiltering(codes=codes)
            for mapping in self._mappings.values():
                if any(cpf.match(out_codes)
                       for out_codes in mapping.out_codes):

                    yield mapping

        else:
            yield from self._mappings.values()

    def get_mappings(
                self,
                codes: list[CodesNSLCE] = None
            ) -> list[CodesMapping]:

        return list(self.iter_mappings(codes))

    def get_mapping(
                self,
                codes: CodesNSLCE
            ) -> CodesMapping:

        return self._mappings[self.grouping.key(codes)]

    def iter_codes(self) -> Generator[CodesNSLCE]:
        for k, mapping in self._mappings.items():
            yield from mapping.out_codes

    def get_codes(self, kind: str = None) -> list[CodesNSLCE]:
        assert kind is None or kind in self.kind_provides
        return list(self.iter_codes())

    def iter_in_codes(
                self,
                mappings: CodesMapping | None = None
            ) -> Generator[CodesNSLCE]:

        for mapping in (mappings or self._mappings).values():
            yield from mapping.in_codes

    def get_in_codes(
                self,
                mappings: CodesMapping | None = None
            ) -> Generator[CodesNSLCE]:

        in_codes = []
        for mapping in mappings or self._mappings:
            in_codes.extend(mapping.in_codes)

        return sorted(in_codes)

    def update_mappings(self) -> None:
        available = None
        for kind in self.kind_requires or [None]:
            codes = set(self._input.get_codes(kind=kind))
            if available is None:
                available = codes
            else:
                available &= codes

        available = sorted(available)
        removed, added = odiff(self._available, available)

        filt = self.filtering.filter
        gkey = self.grouping.key
        mappings = self._mappings

        need_update = set()

        for codes in filt(removed):
            k = gkey(codes)
            mappings[k].in_codes_set.remove(codes)
            need_update.add(k)

        for codes in filt(added):
            k = gkey(codes)
            if k not in mappings:
                mappings[k] = CodesMapping()

            mappings[k].in_codes_set.add(codes)
            need_update.add(k)

        for k in need_update:
            mapping = mappings[k]
            if not mapping.in_codes_set:
                del mappings[k]
            else:
                mapping.in_codes = tuple(sorted(mapping.in_codes_set))
                mapping.out_codes = self.translate_codes(mapping.in_codes)

        self._available = available

    def by_codes(self, xs):
        by_codes = defaultdict(list)
        for x in xs:
            by_codes[x.codes].append(x)

        return by_codes

    def by_codes_unique(self, xs):
        return dict(
            (k, xs_group[0])
            for (k, xs_group) in self.by_codes(xs).items()
            if len(xs_group) == 1)

    def get_channels(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            **kwargs):

        tmin, tmax, codes = get_selection_args(
            CHANNEL, obj, tmin, tmax, time, codes)

        mappings = self.get_mappings(codes)
        in_codes = self.get_in_codes(mappings)

        channels_in = self._input.get_channels(
            codes=in_codes, tmin=tmin, tmax=tmax, **kwargs)

        codes_to_channel = self.by_codes_unique(channels_in)

        channels = []
        for mapping in mappings:
            for in_codes, out_codes in zip(
                    mapping.in_codes, mapping.out_codes):

                channel_in = codes_to_channel[in_codes]
                channel = clone(channel_in)
                channel.codes = out_codes

                channels.append(channel)

        return channels

    def get_time_padding(self):
        return 0.0

    def get_waveforms(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            **kwargs):

        tmin, tmax, codes = get_selection_args(
            CHANNEL, obj, tmin, tmax, time, codes)

        mappings = self.get_mappings(codes)
        in_codes = self.get_in_codes(mappings)

        tpad = self.get_time_padding()

        tmin_raw = tmin - tpad
        tmax_raw = tmax + tpad

        trs = self._input.get_waveforms(
            codes=in_codes, tmin=tmin_raw, tmax=tmax_raw, **kwargs)

        return self.process_waveforms(trs, mappings, in_codes, tmin, tmax)


class Parameters(Object):
    pass


class RestitutionParameters(Parameters):
    frequency_min = Float.T()
    frequency_max = Float.T()
    frequency_taper_factor = Float.T(default=1.5)
    time_taper_factor = Float.T(default=2.0)


class Restitution(Operator):
    translation = AddSuffixTranslation(suffix='R{quantity}')
    quantity = QuantityType.T(default='displacement')

    kind_provides = ('channel', 'waveform')
    kind_requires = ('response',)

    @property
    def name(self):
        return 'Restitution(%s)' % self.quantity[0]

    def translate_codes(self, in_codes):
        return [
            codes.__class__(self.translation.translate(codes).safe_str.format(
                quantity=self.quantity[0]))
            for codes in in_codes]

    def get_time_padding(self):
        return self._parameters.time_taper_factor \
            / self._parameters.frequency_min

    def get_responses_mapping(self, in_codes, tmin, tmax):
        responses = self._input.get_responses(
            codes=in_codes,
            tmin=tmin,
            tmax=tmax)

        return self.by_codes_unique(responses)

    def process_waveforms(self, trs, mappings, in_codes, tmin, tmax):

        tmin_trs = min(tr.tmin for tr in trs)
        tmax_trs = max(tr.tmax for tr in trs)

        codes_to_response = self.get_responses_mapping(
            in_codes, tmin_trs, tmax_trs)

        trs_rest = []
        for tr in trs:
            resp = codes_to_response[tr.codes].get_effective(self.quantity)

            parameters = self._parameters

            freqlimits = (
                parameters.frequency_min / parameters.frequency_taper_factor,
                parameters.frequency_min,
                parameters.frequency_max,
                parameters.frequency_max * parameters.frequency_taper_factor)

            tr_rest = tr.transfer(
                tfade=self.get_time_padding(),
                freqlimits=freqlimits,
                transfer_function=resp,
                invert=True)

            tr_rest.set_codes(*self.get_mapping(tr.codes).out_codes[0])
            trs_rest.append(tr_rest)

        return trs_rest


class Shift(Operator):
    translation = AddSuffixTranslation(suffix='S')
    delay = Duration.T()


class Transform(Operator):
    grouping = Grouping.T(default=SensorGrouping.D())
    translation = ReplaceComponentTranslation(suffix='T{system}')

    kind_provides = ('channel', 'waveform')
    kind_requires = ('channel',)

    def translate_codes(self, in_codes):
        proto = in_codes[0]
        return [
            proto.__class__(
                self.translation.translate(proto).safe_str.format(
                    component=c, system=self.components.lower()))
            for c in self.components]

    def get_channels_mapping(self, in_codes, tmin, tmax):
        channels = self._input.get_channels(
            codes=in_codes,
            tmin=tmin,
            tmax=tmax)

        return self.by_codes_unique(channels)

    def process_waveforms(self, trs, mappings, in_codes, tmin, tmax):

        tmin_trs = min(tr.tmin for tr in trs)
        tmax_trs = max(tr.tmax for tr in trs)

        codes_to_channels = self.get_channels_mapping(
            in_codes, tmin_trs, tmax_trs)

        codes_to_traces = self.by_codes(trs)

        trs_out = []
        for mapping in mappings:
            sensor = Sensor.from_channels_single([
                codes_to_channels[in_codes]
                for in_codes in mapping.in_codes])

            trs_sensor = []
            for in_codes in mapping.in_codes:
                trs_sensor.extend(codes_to_traces[in_codes])

            codes_mapping = dict(zip(self.components, mapping.out_codes))

            trs_sensor_out = self.project(sensor, trs_sensor)
            for tr in trs_sensor_out:
                tr.set_codes(*codes_mapping[tr.channel[-1]])

            trs_out.extend(trs_sensor_out)

        return trs_out


class ToENZ(Transform):
    components = 'ENZ'

    def project(self, sensor, trs_sensor):
        return sensor.project_to_enz(trs_sensor)


class ToTRZ(Transform):
    components = 'TRZ'
    origin = Location.T()

    def project(self, sensor, trs_sensor):
        return sensor.project_to_trz(self.origin, trs_sensor)


class ToLTQ(Transform):
    components = 'LTQ'


class Composition(Operator):
    g = Operator.T()
    f = Operator.T()

    def __init__(self, g, f, **kwargs):
        Operator.__init__(self, g=g, f=f, **kwargs)

    @property
    def name(self):
        return '(%s ○ %s)' % (self.g.name, self.f.name)


__all__ = [
    'Grouping',
    'RegexGrouping',
    'NetworkGrouping',
    'StationGrouping',
    'LocationGrouping',
    'SensorGrouping',
    'ChannelGrouping',
    'Operator',
    'RestitutionParameters',
    'Restitution',
    'Shift',
    'ToENZ',
    'ToTRZ',
    'ToLTQ',
    'Composition']
