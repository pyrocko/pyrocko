# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import annotations

import math
import uuid

from typing import TYPE_CHECKING, Generator, Sequence, Hashable, TypeVar, Union
from typing import List as tList, Set as tSet
from typing import Tuple as tTuple

if TYPE_CHECKING:
    from ..base import Squirrel

from collections import defaultdict
import logging
import re
from itertools import chain

from pyrocko.model import Location
from pyrocko.trace import Trace, TraceTooShort, NoData

from ..model import (
    QuantityType, CodesNSLCE, CodesMatcher, CHANNEL, WAVEFORM,
    get_selection_args, Sensor, Channel, Response, Coverage, join_coverages,
    codes_patterns_list
)

from pyrocko.guts import Object, String, Duration, Float, clone, List

ichain = chain.from_iterable


def lchain(it):
    return list(ichain(it))


HasCodes = TypeVar('HasCodes')
HasTimeAndCodes = TypeVar('HasTimeAndCodes')
TimeFloat = TypeVar('TimeFloat')
CodesConvertible \
    = Union[CodesNSLCE, tList[CodesNSLCE], str, tList[str], tTuple[str],
            tList[tTuple[str]]]

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


def time_min_max(codes_to_traces):
    trs = lchain(codes_to_traces.values())
    tmin_trs = min(tr.tmin for tr in trs)
    tmax_trs = max(tr.tmax for tr in trs)
    return tmin_trs, tmax_trs


def by_codes(xs: HasCodes) -> dict[CodesNSLCE, tList[HasCodes]]:
    by_codes = defaultdict(list)
    for x in xs:
        by_codes[x.codes].append(x)

    return by_codes


def by_codes_unique(xs: HasCodes) -> dict[CodesNSLCE, HasCodes]:
    return dict(
        (k, xs_group[0])
        for (k, xs_group) in by_codes(xs).items()
        if len(xs_group) == 1)


class Filtering(Object):
    '''
    Base class for :py:class:`pyrocko.squirrel.model.Nut` filters.
    '''

    def filter(self, it: Sequence[CodesNSLCE]) -> List[CodesNSLCE]:
        return list(it)


class RegexFiltering(Filtering):
    '''
    Filter by regex.
    '''
    pattern = String.T(default=r'(.*)')

    def __init__(self, **kwargs):
        Filtering.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def filter(self, it: Sequence[CodesNSLCE]) -> List[CodesNSLCE]:
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

    def filter(self, it: Sequence[CodesNSLCE]) -> List[CodesNSLCE]:
        if self._matcher is None:
            return list(it)
        else:
            return list(self._matcher.filter(it))


class Grouping(Object):
    '''
    Base class for :py:class:`pyrocko.squirrel.model.Nut` grouping mechanisms.
    '''

    def key(self, codes: CodesNSLCE) -> Hashable:
        return codes


class RegexGrouping(Grouping):
    '''
    Group by regex pattern.
    '''
    pattern = String.T(default=r'(.*)')

    def __init__(self, **kwargs):
        Grouping.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def key(self, codes: CodesNSLCE) -> Hashable:
        return self._compiled_pattern.fullmatch(codes.safe_str).groups()


class AllGrouping(Grouping):
    def key(self, codes: CodesNSLCE) -> Hashable:
        return None


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


class ComponentGrouping(RegexGrouping):
    '''
    Group by component code.

    All matching channels with the same component code are put into one group.
    '''
    pattern = String.T(default=_cglob_translate('*.*.*.*(?).*'))


class Translation(Object):
    '''
    Base class for :py:class:`pyrocko.squirrel.model.Nut` translators.
    '''

    def translate(self, codes: CodesNSLCE) -> CodesNSLCE:
        return codes


class AddSuffixTranslation(Translation):
    '''
    Add a suffix to :py:attr:`~pyrocko.squirrel.model.CodesNSLCEBase.extra`.
    '''
    suffix = String.T(default='')

    def translate(self, codes: CodesNSLCE) -> CodesNSLCE:
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

    def translate(self, codes: CodesNSLCE) -> CodesNSLCE:
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


class KeepComponentTranslation(RegexTranslation):
    '''
    '''
    pattern = String.T(default=_cglob_translate('*.*.*.*(?).*'))
    replacement = String.T(default=r'...\1.')


class CodesMapping:
    __slots__ = ['in_codes_set', 'in_codes', 'out_codes']

    def __init__(self):
        self.in_codes_set = set()
        self.in_codes = ()
        self.out_codes = ()


def new_operator_id():
    return uuid.uuid4()


class BaseOperator(Object):

    def post_init(self):
        self.reset()

    def reset(self):
        self._operator_id = new_operator_id()
        self._input = None
        self._input_mapping_counter = None
        self._mapping_counter = 0
        self._mappings = {}
        self._available = set()
        self._n_choppers_active = 0

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def kind_provides(self):
        return ('channel', 'response', 'waveform')

    @property
    def kind_requires(self):
        return ()

    def set_input(self, input: Operator | Squirrel) -> None:
        self.reset()
        self._input = input

    def get_input(self) -> (Operator | Squirrel):
        return self._input

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

    def iter_mappings(
                self,
                codes: tList[CodesNSLCE] | None = None
            ) -> Generator[CodesMapping]:

        self.update_mappings()

        if codes is not None:
            if len(codes) == 0:
                return

            cpf = CodesPatternFiltering(codes=codes)

            for mapping in self._mappings.values():
                if any(cpf.match(out_codes)
                       for out_codes in mapping.out_codes):

                    yield mapping
        else:
            yield from self._mappings.values()

    def get_mappings(
                self,
                codes: tList[CodesNSLCE] = None
            ) -> tList[CodesMapping]:

        return list(self.iter_mappings(codes))

    def get_mappings_and_matching_codes(
                self,
                codes: tList[CodesNSLCE] | None = None
                ) -> (tList[CodesMapping], tSet[CodesNSLCE]):

        self.update_mappings()

        if codes is not None:
            if len(codes) == 0:
                return [], set()

            mappings_match = []
            codes_out_match = []
            cpf = CodesPatternFiltering(codes=codes)
            for mapping in self._mappings.values():
                codes_match_this = [
                    out_codes for out_codes in mapping.out_codes
                    if cpf.match(out_codes)]

                if codes_match_this:
                    codes_out_match.extend(codes_match_this)
                    mappings_match.append(mapping)

            return mappings_match, codes_out_match
        else:
            return list(self._mappings.values()), None

    def iter_codes(self) -> Generator[CodesNSLCE]:

        self.update_mappings()

        for mapping in self._mappings.values():
            yield from mapping.out_codes

    def get_codes(self, kind: str = None) -> tList[CodesNSLCE]:
        assert kind is None or kind in self.kind_provides
        return list(self.iter_codes())

    def iter_in_codes(
                self,
                mappings: tList[CodesMapping] | None = None
            ) -> Generator[CodesNSLCE]:

        self.update_mappings()

        if mappings is None:
            mappings = self._mappings.values()

        for mapping in mappings:
            yield from mapping.in_codes

    def get_in_codes(
                self,
                mappings: tList[CodesMapping] | None = None
            ) -> Generator[CodesNSLCE]:

        return sorted(self.iter_in_codes(mappings))

    def update_mappings(self) -> None:

        if isinstance(self._input, Operator):
            self._input.update_mappings()

        if self._input._mapping_counter == self._input_mapping_counter:
            return

        available = None
        for kind in self.kind_requires or [None]:
            codes = set(self._input.get_codes(kind=kind))
            if available is None:
                available = codes
            else:
                available &= codes

        added = available - self._available
        removed = self._available - available

        need_update = self._update_mappings_specific(added, removed)
        if not isinstance(need_update, bool):
            raise Exception(
                '_update_mappings_specific(...) must return bool')

        self._input_mapping_counter = self._input._mapping_counter
        if need_update:
            self._mapping_counter += 1

    def get_in_channels(
                self,
                in_codes: tList[CodesNSLCE],
                tmin: TimeFloat,
                tmax: TimeFloat,
            ) -> dict[CodesNSLCE, Channel]:

        channels = self._input.get_channels(
            codes=in_codes,
            tmin=tmin,
            tmax=tmax)

        return by_codes(channels)

    def get_in_responses(
                self,
                in_codes: tList[CodesNSLCE],
                tmin: TimeFloat,
                tmax: TimeFloat,
            ) -> dict[CodesNSLCE, Response]:

        responses = self._input.get_responses(
            codes=in_codes,
            tmin=tmin,
            tmax=tmax)

        return by_codes(responses)

    def get_in_waveforms(
                self,
                in_codes: tList[CodesNSLCE],
                tmin: TimeFloat,
                tmax: TimeFloat,
                **kwargs,
            ) -> dict[CodesNSLCE, Trace]:

        traces = self._input.get_waveforms(
            codes=in_codes, tmin=tmin, tmax=tmax, **kwargs)

        return by_codes(traces)

    def get_in_coverages(
                self,
                kind: str,
                in_codes: tList[CodesNSLCE],
                tmin: TimeFloat,
                tmax: TimeFloat,
                **kwargs,
            ) -> dict[CodesNSLCE, Trace]:

        coverages = self._input.get_coverage(
            kind, codes=in_codes, tmin=tmin, tmax=tmax, **kwargs)

        return by_codes_unique(coverages)

    def get_channels(
                self,
                obj: HasTimeAndCodes = None,
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
                time: TimeFloat = None,
                codes: CodesConvertible = None,
            ) -> tList[Channel]:

        tmin, tmax, codes = get_selection_args(
            CHANNEL, obj, tmin, tmax, time, codes)

        mappings, codes_want = self.get_mappings_and_matching_codes(codes)
        in_codes = self.get_in_codes(mappings)
        channels_in = self.get_in_channels(in_codes, tmin, tmax)

        channels_out = self.process_channels(
            mappings, in_codes, channels_in, tmin, tmax)

        if codes_want is not None:
            channels_out = [
                channel for channel in channels_out
                if channel.codes in codes_want]

        return channels_out

    def get_sensors(
                self,
                obj: HasTimeAndCodes = None,
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
                time: TimeFloat = None,
                codes: CodesConvertible = None,
            ) -> tList[Sensor]:

        channels = self.get_channels(
            obj=obj,
            tmin=tmin,
            tmax=tmax,
            time=time,
            codes=codes)

        return self.process_sensors(channels)

    def get_responses(
                self,
                obj: HasTimeAndCodes = None,
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
                time: TimeFloat = None,
                codes: CodesConvertible = None,
            ) -> tList[Response]:

        tmin, tmax, codes = get_selection_args(
            CHANNEL, obj, tmin, tmax, time, codes)

        mappings = self.get_mappings(codes)
        in_codes = self.get_in_codes(mappings)
        codes_to_responses = self.get_in_responses(in_codes, tmin, tmax)
        return self.process_responses(
            mappings, in_codes, codes_to_responses, tmin, tmax)

    def get_waveforms(
                self,
                obj: HasTimeAndCodes = None,
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
                time: TimeFloat = None,
                codes: CodesConvertible = None,
                **kwargs
            ) -> tList[Trace]:

        tmin, tmax, codes = get_selection_args(
            CHANNEL, obj, tmin, tmax, time, codes)

        mappings, codes_want = self.get_mappings_and_matching_codes(codes)
        in_codes = self.get_in_codes(mappings)

        tpad = self.get_time_padding()

        in_tmin = tmin - tpad
        in_tmax = tmax + tpad

        codes_to_traces = self.get_in_waveforms(
            in_codes, in_tmin, in_tmax, **kwargs)

        if not codes_to_traces:
            return []

        trs = self.process_waveforms(
            mappings, in_codes, codes_to_traces, tmin, tmax)

        if codes_want is not None:
            trs = [tr for tr in trs if tr.codes in codes_want]

        return trs

    def get_squirrel(self):
        from ..base import Squirrel
        if self._input is None or isinstance(self._input, Squirrel):
            return self._input
        else:
            return self._input.get_squirrel()

    def chopper_waveforms(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            codes_exclude=None, sample_rate_min=None, sample_rate_max=None,
            tinc=None, tpad=0.,
            want_incomplete=True, snap_window=False,
            degap=True, maxgap=5, maxlap=None,
            snap=None, include_last=False, load_data=True,
            accessor_id=None, clear_accessor=True,   # operator_params=None,
            grouping=None, channel_priorities=None):

        from ..base import Batch

        tmin, tmax, codes = get_selection_args(
            WAVEFORM, obj, tmin, tmax, time, codes)

        kinds = ['waveform']
        self_tmin, self_tmax = self.get_time_span(kinds, dummy_limits=False)

        if None in (self_tmin, self_tmax):
            logger.warning(
                'Content has undefined time span. No waveforms and no '
                'waveform promises?')
            return

        if snap_window and tinc is not None:
            tmin = tmin if tmin is not None else self_tmin
            tmax = tmax if tmax is not None else self_tmax
            tmin = math.floor(tmin / tinc) * tinc
            tmax = math.ceil(tmax / tinc) * tinc
        else:
            tmin = tmin if tmin is not None else self_tmin + tpad
            tmax = tmax if tmax is not None else self_tmax - tpad

        if tinc is None:
            tinc = tmax - tmin
            nwin = 1
        elif tinc == 0.0:
            nwin = 1
        else:
            eps = 1e-6
            nwin = max(1, int((tmax - tmin) / tinc - eps) + 1)

        try:
            if accessor_id is None:
                accessor_id = 'chopper_%s_%i' % (
                    str(self._operator_id),
                    self._n_choppers_active)

            self._n_choppers_active += 1

            if grouping is None:
                codes_list = [codes]
            else:
                operator = Operator(
                    filtering=CodesPatternFiltering(codes=codes),
                    grouping=grouping)

                operator.set_input(self)

                codes_list = [
                    codes_patterns_list(mapping.in_codes)
                    for mapping in operator.iter_mappings()]

            ngroups = len(codes_list)
            for igroup, scl in enumerate(codes_list):
                for iwin in range(nwin):
                    wmin, wmax = tmin+iwin*tinc, min(tmin+(iwin+1)*tinc, tmax)

                    chopped = self.get_waveforms(
                        tmin=wmin-tpad,
                        tmax=wmax+tpad,
                        codes=scl,
                        codes_exclude=codes_exclude,
                        sample_rate_min=sample_rate_min,
                        sample_rate_max=sample_rate_max,
                        snap=snap,
                        include_last=include_last,
                        load_data=load_data,
                        want_incomplete=want_incomplete,
                        degap=degap,
                        maxgap=maxgap,
                        maxlap=maxlap,
                        accessor_id=accessor_id,
                        channel_priorities=channel_priorities)

                    self.get_squirrel().advance_accessor(accessor_id)

                    yield Batch(
                        tmin=wmin,
                        tmax=wmax,
                        tpad=tpad,
                        i=iwin,
                        n=nwin,
                        igroup=igroup,
                        ngroups=ngroups,
                        traces=chopped)

        finally:
            self._n_choppers_active -= 1
            if clear_accessor:
                self.get_squirrel().clear_accessor(accessor_id, 'waveform')

    def get_coverage(
            self,
            kind,
            tmin: TimeFloat = None,
            tmax: TimeFloat = None,
            codes: CodesConvertible = None,
            limit=None) -> tList[Coverage]:

        tmin, tmax, codes = get_selection_args(
            CHANNEL, None, tmin, tmax, None, codes)

        mappings = self.get_mappings(codes)
        in_codes = self.get_in_codes(mappings)

        tpad = self.get_time_padding()

        in_tmin = tmin - tpad if tmin is not None else None
        in_tmax = tmax + tpad if tmax is not None else None

        codes_to_coverage = self.get_in_coverages(
            kind, in_codes, in_tmin, in_tmax)

        return self.process_coverage(
            mappings, in_codes, codes_to_coverage, tmin, tmax)

    def process_channels(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_channels: dict[CodesNSLCE, tList[Channel]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Channel]:

        channels = []
        for mapping in mappings:
            for in_codes, out_codes in zip(
                    mapping.in_codes, mapping.out_codes):

                if in_codes not in codes_to_channels:
                    # print('not available: %s' % in_codes.safe_str)
                    continue

                for channel_in in codes_to_channels[in_codes]:
                    channel = clone(channel_in)
                    channel.codes = out_codes
                    channels.append(channel)

        return channels

    def process_sensors(
                self,
                channels: tList[Channel],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Sensor]:

        return Sensor.from_channels(channels)

    def process_responses(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_responses: dict[CodesNSLCE, tList[Response]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Response]:

        responses = []
        for mapping in mappings:
            for in_codes, out_codes in zip(
                    mapping.in_codes, mapping.out_codes):

                try:
                    responses_in = codes_to_responses[in_codes]
                except KeyError:
                    continue

                for response_in in responses_in:
                    response = clone(response_in)
                    response.codes = out_codes
                    responses.append(response)

        return responses

    def process_waveforms(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_traces: dict[CodesNSLCE, Trace],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        return lchain(codes_to_traces.values())

    def process_coverage(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_coverage: dict[CodesNSLCE, Coverage],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        coverages = []
        for mapping in mappings:
            coverages_group = [
                codes_to_coverage[in_codes]
                for in_codes in mapping.in_codes
                if in_codes in codes_to_coverage]

            if not coverages_group:
                continue

            coverage_common = join_coverages(
                coverages_group,
                tbleed=self.get_time_padding())

            for out_codes in mapping.out_codes:
                coverages.append(Coverage(
                    kind_id=coverage_common.kind_id,
                    codes=out_codes,
                    tmin=coverage_common.tmin,
                    tmax=coverage_common.tmax,
                    deltat=coverage_common.deltat,
                    changes=coverage_common.changes))

        return coverages

    def get_time_padding(self) -> float:
        return 0.0

    def get_time_span(
            self,
            kinds,
            dummy_limits=True) -> tTuple[TimeFloat, TimeFloat]:

        tmin, tmax = self._input.get_time_span(
            kinds, dummy_limits=dummy_limits)

        if None in (tmin, tmax):
            return (None, None)

        return tmin + self.get_time_padding(), tmax - self.get_time_padding()


class Operator(BaseOperator):
    '''
    Base class for operators with typical filter-group-translate behaviour.
    '''

    filtering = Filtering.T(default=Filtering.D())
    grouping = Grouping.T(default=Grouping.D())
    translation = Translation.T(default=Translation.D())

    def translate_codes(
                self,
                in_codes: tList[CodesNSLCE]
            ) -> tList[CodesNSLCE]:
        return [self.translation.translate(codes) for codes in in_codes]

    def _update_mappings_specific(
            self,
            added: tSet[CodesNSLCE],
            removed: tSet[CodesNSLCE]) -> bool:

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
            mapping = self._mappings[k]
            if not mapping.in_codes_set:
                del self._mappings[k]
            else:
                mapping.in_codes = tuple(sorted(mapping.in_codes_set))
                mapping.out_codes = self.translate_codes(mapping.in_codes)

        return bool(need_update)


class Restitution(Operator):
    translation = AddSuffixTranslation(suffix='R{quantity}')
    quantity = QuantityType.T(default='velocity')
    frequency_min = Float.T()
    frequency_max = Float.T()
    frequency_taper_factor = Float.T(default=1.5)
    frequency_taper_min = Float.T(optional=True)
    frequency_taper_max = Float.T(optional=True)
    time_taper_factor = Float.T(default=2.0)
    time_padding_extra_factor = Float.T(default=2.0)

    @property
    def kind_provides(self):
        return ('channel', 'waveform')

    @property
    def kind_requires(self):
        return ('response',)

    @property
    def name(self) -> str:
        return 'Restitution(%s)' % self.quantity[0]

    def translate_codes(
            self,
            in_codes: tList[CodesNSLCE]) -> tList[CodesNSLCE]:

        return [
            codes.__class__(self.translation.translate(codes).safe_str.format(
                quantity=self.quantity[0]))
            for codes in in_codes]

    def get_time_padding(self) -> float:
        return self.time_taper_factor \
            / self.frequency_min * self.time_padding_extra_factor

    def get_taper_frequencies(self):
        return (
            self.frequency_min / self.frequency_taper_factor
            if self.frequency_taper_min is None else self.frequency_taper_min,
            self.frequency_min,
            self.frequency_max,
            self.frequency_max * self.frequency_taper_factor
            if self.frequency_taper_max is None else self.frequency_taper_max)

    def process_waveforms(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_traces: dict[CodesNSLCE, Trace],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        tmin_trs, tmax_trs = time_min_max(codes_to_traces)

        codes_to_responses = self.get_in_responses(
            in_codes, tmin_trs, tmax_trs)

        freqlimits = self.get_taper_frequencies()

        traces_out = []
        for mapping in mappings:
            for in_codes, out_codes in zip(
                    mapping.in_codes, mapping.out_codes):

                if in_codes not in codes_to_responses:
                    logger.warning(
                        'No instrument response available: %s' % str(in_codes))
                    continue

                responses = codes_to_responses[in_codes]
                if len(responses) == 0:
                    logger.warning(
                        'No instrument response available: %s' % str(in_codes))
                    continue

                elif len(responses) > 1:
                    logger.warning(
                        'Multiple matching instrument responses: %s'
                        % str(in_codes))
                    continue

                resp = responses[0].get_effective(self.quantity)

                for tr in codes_to_traces[in_codes]:
                    if freqlimits[-1] > 0.5/tr.deltat:
                        print(
                            'sampling rate too low for restitution frequency '
                            'range: %s' % tr.summary)
                        continue

                    try:
                        # ymean = int(num.mean(
                        #     tr.chop(tmin, tmax, inplace=False).ydata))
                        tr.ydata = tr.ydata.astype(float)
                        # tr.ydata -= ymean
                        tr_rest = tr.transfer(
                            tfade=self.time_taper_factor / self.frequency_min,
                            freqlimits=freqlimits,
                            transfer_function=resp,
                            # demean=True,
                            invert=True)

                        # tr_rest.ydata += ymean

                        tr_rest.set_codes(*out_codes)
                        tr_rest.chop(tmin, tmax)
                        traces_out.append(tr_rest)

                    except (TraceTooShort, NoData):
                        # print('trace too short: %s' % tr.summary)
                        pass

        return traces_out


class Shift(Operator):
    translation = AddSuffixTranslation(suffix='S')
    delay = Duration.T()


class Transform(Operator):
    grouping = Grouping.T(default=SensorGrouping.D())
    translation = ReplaceComponentTranslation(suffix='T{system}')

    @property
    def kind_provides(self):
        return ('channel', 'waveform')

    @property
    def kind_requires(self):
        return ('channel',)

    def translate_codes(
            self,
            in_codes: tList[CodesNSLCE]) -> tList[CodesNSLCE]:

        proto = in_codes[0]
        return [
            proto.__class__(
                self.translation.translate(proto).safe_str.format(
                    component=c, system=self.components.lower()))
            for c in self.components]

    def process_channels(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_channels: dict[CodesNSLCE, tList[Channel]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Channel]:

        channels_out = []
        for mapping in mappings:
            channels = []
            for in_codes in mapping.in_codes:
                try:
                    channels.extend(codes_to_channels[in_codes])
                except KeyError:
                    pass

            sensors = Sensor.from_channels(channels)
            for sensor in sensors:
                for component, out_codes in zip(
                        self.components, mapping.out_codes):

                    channel = clone(sensor.channels[0])
                    channel.codes = out_codes
                    channel.azimuth, channel.dip = self.get_orientation(
                        sensor, component)

                    channels_out.append(channel)

        return channels_out

    def process_waveforms(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_traces: dict[CodesNSLCE, Trace],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        tmin_trs, tmax_trs = time_min_max(codes_to_traces)

        codes_to_channels = self.get_in_channels(
            in_codes, tmin_trs, tmax_trs)

        traces_out = []
        for mapping in mappings:
            channels = []
            for in_codes in mapping.in_codes:
                try:
                    channels.extend(codes_to_channels[in_codes])
                except KeyError:
                    pass

            sensors = Sensor.from_channels(channels)
            if len(sensors) == 0:
                logger.warning(
                    'No matching sensors: %s', ', '.join(
                        c.safe_str for c in mapping.in_codes))

                continue

            if len(sensors) != 1:
                logger.warning(
                    'Multiple matching sensors: %s'
                    % ', '.join(sensor.codes.safe_str for sensor in sensors))

                continue

            for sensor in sensors:
                trs_sensor = []
                for in_codes in mapping.in_codes:
                    trs_sensor.extend(codes_to_traces[in_codes])

                codes_mapping = dict(zip(self.components, mapping.out_codes))

                trs_sensor_out = self.project(sensor, trs_sensor)
                for tr in trs_sensor_out:
                    tr.set_codes(*codes_mapping[tr.channel[-1]])

                traces_out.extend(trs_sensor_out)

        return traces_out


class ToENZ(Transform):
    components = 'ENZ'

    def project(self, sensor, trs_sensor):
        return sensor.project_to_enz(trs_sensor)

    def get_orientation(self, sensor, component):
        return {
            'E': (90., 0.),
            'N': (0., 0.),
            'Z': (0., -90.)}[component]


class ToTRZ(Transform):
    components = 'TRZ'
    origin = Location.T()

    def project(self, sensor, trs_sensor):
        return sensor.project_to_trz(self.origin, trs_sensor)

    def get_orientation(self, sensor, component):
        _, bazi = self.origin.azibazi_to(sensor)

        return {
            'T': ((bazi + 270. + 180.) % 360. - 180., 0.),
            'R': ((bazi + 180. + 180.) % 360. - 180., 0.),
            'Z': (0, -90.)}[component]


__all__ = [
    'CodesConvertible',
    'HasTimeAndCodes',
    'Filtering',
    'RegexFiltering',
    'CodesPatternFiltering',
    'Grouping',
    'AllGrouping',
    'RegexGrouping',
    'NetworkGrouping',
    'StationGrouping',
    'LocationGrouping',
    'SensorGrouping',
    'ChannelGrouping',
    'ComponentGrouping',
    'Translation',
    'AddSuffixTranslation',
    'RegexTranslation',
    'ReplaceComponentTranslation',
    'KeepComponentTranslation',
    'BaseOperator',
    'Operator',
    'Restitution',
    'Shift',
    'ToENZ',
    'ToTRZ',
]
