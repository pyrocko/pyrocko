# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import annotations

import math
import uuid

from typing import TYPE_CHECKING, Generator, Sequence, TypeVar, Union
from typing import List as tList, Set as tSet
from typing import Tuple as tTuple

if TYPE_CHECKING:
    from ..base import Squirrel

from collections import defaultdict
import logging
from itertools import chain

from pyrocko import util
from pyrocko.model import Location, Event
from pyrocko.trace import Trace, TraceTooShort, NoData
from pyrocko.carpet import Carpet, sum as sum_carpets, CarpetError
from pyrocko.response import InvalidResponseError
from pyrocko.gf import Earthmodel1D

from ..model import (
    QuantityType, CodesNSLCE, CodesMatcher, CHANNEL, WAVEFORM, CARPET,
    get_selection_args, Sensor, Channel, Response, Coverage, join_coverages,
    codes_patterns_list, make_rich_coverage
)

from ..error import SquirrelError

from pyrocko.guts import (
    Object, String, Duration, Float, Bool, clone, List, Dict, equal
)

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


class OperatorError(SquirrelError):
    pass


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
            cs[0] if all(c == cs[0] for c in cs) else '{%s}' % ','.join(cs)
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


class CodesFilterBase(Object):
    '''
    Base class for :py:class:`pyrocko.squirrel.model.Nut` filters.
    '''

    __eq__ = equal

    def filter(self, it: Sequence[CodesNSLCE]) -> List[CodesNSLCE]:
        return list(it)


class CodesFilter(CodesFilterBase):
    '''
    Filter by codes patterns.
    '''
    include = List.T(CodesNSLCE.T(), optional=True)
    exclude = List.T(CodesNSLCE.T(), optional=True)

    def __init__(self, **kwargs):
        CodesFilterBase.__init__(self, **kwargs)
        if self.include is not None:
            self._matcher = CodesMatcher(self.include)
        else:
            self._matcher = None

        if self.exclude is not None:
            self._matcher_exclude = CodesMatcher(self.exclude)
        else:
            self._matcher_exclude = None

    def match(self, codes):
        return (self._matcher is None or self._matcher.match(codes)) \
            and (self.exclude is None
                 or not self._matcher_exclude.match(codes))

    def filter(self, it: Sequence[CodesNSLCE]) -> List[CodesNSLCE]:
        if self._matcher is None and self._matcher_exclude is None:
            return list(it)
        elif self._matcher_exclude is None:
            return list(self._matcher.filter(it))
        else:
            return [codes for codes in it if self.match(codes)]


class CodesMapping:
    __slots__ = ['in_codes_set', 'in_codes', 'out_codes', 'group_key']

    def __init__(self):
        self.in_codes_set = set()
        self.in_codes = ()
        self.out_codes = ()
        self.group_key = ''

    def describe(self):
        return '%i <- %i' % (len(self.out_codes), len(self.in_codes))


def new_operator_id():
    return uuid.uuid4()


class Outlet(Object):
    kinds = List.T(String.T())
    attributes = Dict.T(String.T(), String.T())


class InputCombinator:

    def __init__(self, inputs):
        for input in inputs:
            self.add_input(input)

    def add_input(self, input):
        if input in self._inputs:
            raise OperatorError(
                f'Input "{input.name}" already among inputs.')

        self._inputs.append(input)

    def reset(self):
        self._input_mapping_counters = None
        self._mapping_counter = 0

    def update_mappings(self):
        self._input_mapping_counters = [0] * len(self._inputs)
        need_update = False
        for iinput, input in enumerate(self._inputs):
            input_mapping_counter = input.update_mappings()
            if input_mapping_counter != self._input_mapping_counters[iinput]:
                self._input_mapping_counters[iinput] = input_mapping_counter
                need_update = True

        if need_update:
            self._mapping_counter += 1

        return self._mapping_counter

    def _aggregate(self, method_name, *args, **kwargs):
        xs = []
        for input in self._inputs:
            xs.extend(getattr(input, method_name)(*args, **kwargs))

        return xs

    def get_time_span(
            self,
            kinds,
            dummy_limits=True) -> tTuple[TimeFloat, TimeFloat]:

        tmins, tmaxs = zip(*(
            input.get_time_span(kinds, dummy_limits=dummy_limits)
            for input in self._inputs))

        tmins = [tmin for tmin in tmins if tmin is not None]
        tmaxs = [tmax for tmax in tmaxs if tmax is not None]

        if not tmins or not tmaxs:
            return (None, None)

        return min(tmins), max(tmaxs)

    def iter_in_codes(self, *args, **kwargs):
        return iter(
            sorted(set(self._aggregate('get_in_codes', *args, **kwargs))))

    def iter_codes(self, *args, **kwargs):
        return iter(
            sorted(set(self._aggregate('get_codes', *args, **kwargs))))

    def get_in_codes(self, *args, **kwargs):
        return sorted(set(self._aggregate('get_in_codes', *args, **kwargs)))

    def get_codes(self, *args, **kwargs):
        return sorted(set(self._aggregate('get_codes', *args, **kwargs)))

    def get_carpets(self, *args, **kwargs):
        return self._aggregate('get_carpets', *args, **kwargs)

    def get_channels(self, *args, **kwargs):
        return self._aggregate('get_channels', *args, **kwargs)

    def get_events(self, *args, **kwargs):
        return self._aggregate('get_events', *args, **kwargs)

    def get_waveforms(self, *args, **kwargs):
        return self._aggregate('get_waveforms', *args, **kwargs)

    def get_responses(self, *args, **kwargs):
        return self._aggregate('get_responses', *args, **kwargs)

    def get_rich_coverage(self, *args, **kwargs):
        return self._aggregate('get_rich_coverage', *args, **kwargs)

    def get_sensors(self, *args, **kwargs):
        return self._aggregate('get_sensors', *args, **kwargs)

    def get_squirrel(self, *args, **kwargs):
        squirrels = [input.get_squirrel() for input in self._inputs]
        if not all(squirrels[0] is squirrel for squirrel in squirrels):
            raise OperatorError(
                'Currently, only a single root Squirrel is supported.')

        return squirrels[0]

    def advance_accessor(self, *args, **kwargs):
        for input in self._inputs:
            input.advance_accessor(*args, **kwargs)


class BaseOperator(Object):

    name = String.T(default='base_op')
    input_names = List.T(String.T(), optional=True)

    def post_init(self):
        self.reset()
        if self.name is None:
            self.name = self.__class__.__name__

    def reset(self):
        self._operator_id = new_operator_id()
        self._input = None
        self._mantra = None
        self._input_mapping_counter = None
        self._mapping_counter = 0
        self._mappings = {}
        self._available = set()
        self._n_choppers_active = 0

    @property
    def kind_provides(self):
        return ('channel', 'response', 'waveform')

    @property
    def kind_requires(self):
        return ()

    def add_input(self, input: Operator | Squirrel) -> None:
        for kind in self.kind_requires:
            if kind not in input.kind_provides:
                raise Exception(
                    'Operator %s requires "%s" but input operator %s does not '
                    'provide it.' % (self.__class__, kind, input.__class__))

        if self._input is None:
            self._input = input
        elif not isinstance(self._input, InputCombinator):
            self._input = InputCombinator([input, self._input])
        elif isinstance(self._input, InputCombinator):
            self._input.add_input(input)
        else:
            assert False

    def get_input(self) -> (Operator | Squirrel):
        return self._input

    def set_mantra(self, mantra):
        self._mantra = mantra

    @property
    def mantra_name(self):
        return self._mantra.name if self._mantra is not None else ''

    def describe(self) -> str:
        return '''%s:
  provides: %s
  requires: %s
  outlets:
%s
  mappings:
%s''' % (
            self.name,
            ', '.join(self.kind_provides),
            ', '.join(self.kind_requires),
            self._str_outlets,
            self._str_mappings)

    @property
    def _str_outlets(self) -> str:
        lines = []
        for ioutlet, outlet in enumerate(self.get_outlets()):
            lines.append(
                '    %i: %s :: %s' % (
                    ioutlet,
                    ', '.join(outlet.kinds),
                    ', '.join(
                        '%s=%s' % (k, v)
                        for (k, v) in outlet.attributes.items())))

        return '\n'.join(lines)

    @property
    def _str_mappings(self) -> str:
        return '\n'.join([
            '    %s <- %s' % (
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

            cpf = CodesFilter(include=codes)

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
            cpf = CodesFilter(include=codes)
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
        if kind not in self.kind_provides:
            return []
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
            ) -> tList[CodesNSLCE]:

        return sorted(self.iter_in_codes(mappings))

    def update_mappings(self) -> None:

        input_mapping_counter = self._input.update_mappings()
        if input_mapping_counter == self._input_mapping_counter:
            return self._mapping_counter

        available = None
        for kind in self.kind_requires or [None]:
            codes = set(self._input.get_codes(kind=kind))
            if available is None:
                available = codes
            else:
                available &= codes

        added = available - self._available
        removed = self._available - available

        self._available = available

        need_update = self._update_mappings_specific(added, removed)
        assert isinstance(need_update, bool), \
            '_update_mappings_specific(...) must return bool'

        self._input_mapping_counter = input_mapping_counter
        if need_update:
            self._mapping_counter += 1
            logger.debug(
                f'Mapping updated for {self.name}: {self._mapping_counter}')

        return self._mapping_counter

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

    def get_in_carpets(
                self,
                in_codes: tList[CodesNSLCE],
                tmin: TimeFloat,
                tmax: TimeFloat,
                **kwargs,
            ) -> dict[CodesNSLCE, Carpet]:

        carpets = self._input.get_carpets(
            codes=in_codes, tmin=tmin, tmax=tmax, **kwargs)

        return by_codes(carpets)

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

    def get_rich_coverage(
            self, tmin=None, tmax=None, codes=None, limit=None):

        kinds = [
            'channel',
            'response',
            'waveform',
            'waveform_promise',
            'carpet']

        coverages_all = []
        for kind in kinds:
            coverages_all.extend(self.get_coverage(
                kind, tmin=tmin, tmax=tmax, codes=codes, limit=limit))

        coverages_by_codes = util.group_by(
            lambda coverage: coverage.codes, coverages_all)

        def str_codes(codes):
            return ', '.join(c.safe_str for c in codes)

        return [
            make_rich_coverage(kinds, coverages)
            for _, coverages in coverages_by_codes.items()]

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

    def get_carpets(
                self,
                obj: HasTimeAndCodes = None,
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
                time: TimeFloat = None,
                codes: CodesConvertible = None,
                **kwargs
            ) -> tList[Carpet]:

        tmin, tmax, codes = get_selection_args(
            CHANNEL, obj, tmin, tmax, time, codes)

        mappings, codes_want = self.get_mappings_and_matching_codes(codes)
        in_codes = self.get_in_codes(mappings)

        tpad = self.get_time_padding()

        in_tmin = tmin - tpad
        in_tmax = tmax + tpad

        codes_to_carpets = self.get_in_carpets(
            in_codes, in_tmin, in_tmax, **kwargs)

        if not codes_to_carpets:
            return []

        carpets = self.process_carpets(
            mappings, in_codes, codes_to_carpets, tmin, tmax)

        if codes_want is not None:
            carpets = [
                carpet for carpet in carpets if carpet.codes in codes_want]

        return carpets

    def chopper_carpets(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            tinc=None, tpad=0., want_incomplete=True, snap_window=False):

        tmin, tmax, codes = get_selection_args(
            CARPET, obj, tmin, tmax, time, codes)

        tmin_content, tmax_content = self.get_time_span(['carpet'])

        source_gen = util.iter_windows(
                    tmin=tmin,
                    tmax=tmax,
                    tinc=tinc,
                    tpad=tpad,
                    snap_window=snap_window,
                    tmin_content=tmin_content,
                    tmax_content=tmax_content)

        def gen():
            for batch in source_gen:
                carpets = self.get_carpets(
                    tmin=batch.tmin-tpad, tmax=batch.tmax+tpad, codes=codes)
                batch.carpets = carpets
                yield batch

        return util.GeneratorWithLen(gen(), len(source_gen))

    def get_events(
                self,
                obj: HasTimeAndCodes = None,
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
                time: TimeFloat = None,
                codes: CodesConvertible = None,
                **kwargs
            ) -> tList[Event]:

        return []

    def get_squirrel(self):
        from ..base import Squirrel
        if self._input is None or isinstance(self._input, Squirrel):
            return self._input
        else:
            return self._input.get_squirrel()

    def advance_accessor(self, accessor_id='default', cache_id=None):
        self._input.advance_accessor(accessor_id, cache_id)

    def chopper_waveforms(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            codes_exclude=None, sample_rate_min=None, sample_rate_max=None,
            tinc=None, tpad=0.,
            want_incomplete=True, snap_window=False,
            degap=True, maxgap=5, maxlap=None,
            snap=None, include_last=False, load_data=True,
            accessor_id=None, clear_accessor=True,   # operator_params=None,
            group_by=None, channel_priorities=None):

        from ..base import Batch

        tmin, tmax, codes = get_selection_args(
            WAVEFORM, obj, tmin, tmax, time, codes)

        kinds = ['waveform', 'waveform_promise']
        self_tmin, self_tmax = self.get_time_span(kinds, dummy_limits=False)

        if None in (self_tmin, self_tmax) and (tmin is None and tmax is None):
            logger.warning(
                'Content has undefined time span. No waveforms and no '
                'waveform promises?')
            return []

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

            if group_by is None:
                codes_list = [codes]
            else:
                operator = Operator(
                    codes_projection=CodesProjection(
                        include=codes,
                        group_by=group_by))

                operator.set_input(self)

                codes_list = [
                    codes_patterns_list(mapping.in_codes)
                    for mapping in operator.iter_mappings()]

            ngroups = len(codes_list)

            def gen():
                for igroup, scl in enumerate(codes_list):
                    for iwin in range(nwin):
                        wmin, wmax = tmin+iwin*tinc, min(
                            tmin+(iwin+1)*tinc, tmax)

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

                        self.advance_accessor(accessor_id, 'waveform')

                        yield Batch(
                            tmin=wmin,
                            tmax=wmax,
                            tpad=tpad,
                            i=iwin,
                            n=nwin,
                            igroup=igroup,
                            ngroups=ngroups,
                            traces=chopped)

            return util.GeneratorWithLen(gen(), ngroups*nwin)

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

        coverages = self.process_coverage(
            mappings, in_codes, codes_to_coverage, tmin, tmax)

        if codes is None:
            return coverages

        matcher = CodesMatcher(codes)
        return [
            coverage
            for coverage in coverages
            if matcher.match(coverage.codes)]

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

    def process_carpets(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_carpets: dict[CodesNSLCE, Carpet],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Carpet]:

        return lchain(codes_to_carpets.values())

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

            try:
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

            except NoData:
                continue

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


class EmptyStrings:
    def __getattr__(self, k):
        return ''


empty_strings = EmptyStrings()


class CodesProjectionBase(CodesFilter):

    def group_key(self, codes):
        raise NotImplementedError()

    def project(self, operator, codes, outlets):
        raise NotImplementedError()


class CodesProjection(CodesProjectionBase):

    template = String.T(
        default='{i.network}.{i.station}.{i.location}.{i.channel}.{i.extra}')

    group_by = String.T(optional=True)

    def __init__(self, template=None, group_by=None):
        d = {}
        if template is not None:
            d['template'] = template

        if group_by is not None:
            d['group_by'] = group_by

        CodesProjectionBase.__init__(self, **d)

    def group_key(self, codes):
        return (self.group_by or self.template).format(
            i=codes, o=empty_strings)

    def _project_single(self, operator, codes, outlet):
        d = dict(name=operator.name, mantra=operator.mantra_name)
        d.update(outlet.attributes)
        o = util.Anon(**d)
        return CodesNSLCE(self.template.format(i=codes, o=o))

    def project(self, operator, codes_group, outlets):
        return tuple(sorted(set(
            self._project_single(operator, codes, outlet)
            for codes in codes_group
            for outlet in outlets)))


def basic_codes_projection_t(template):
    return CodesProjectionBase.T(default=CodesProjection.D(template=template))


class Operator(BaseOperator):
    '''
    Base class for operators with typical filter-group-translate behaviour.
    '''

    name = String.T(default='op')
    codes_projection = basic_codes_projection_t(
        '{i.network}.{i.station}.{i.location}.{i.channel}.{i.extra}')

    def get_outlets(self):
        return [Outlet()]

    def _update_mappings_specific(
            self,
            added: tSet[CodesNSLCE],
            removed: tSet[CodesNSLCE]) -> bool:

        filt = self.codes_projection.filter
        gkey = self.codes_projection.group_key
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
                mappings[k].group_key = k

            mappings[k].in_codes_set.add(codes)
            need_update.add(k)

        for k in need_update:
            mapping = self._mappings[k]
            if not mapping.in_codes_set:
                del self._mappings[k]
            else:
                mapping.in_codes = tuple(sorted(mapping.in_codes_set))
                mapping.out_codes = self.codes_projection.project(
                    self, mapping.in_codes, self.get_outlets())

        return bool(need_update)


class Restitution(Operator):
    name = String.T(default='rest')
    codes_projection = basic_codes_projection_t(
        '{i.network}.{i.station}.{i.location}.{i.channel}'
        '.{i.extra}R{o.quantity}')
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
        return ('waveform', 'response')

    def get_outlets(self) -> tList[Outlet]:
        return [Outlet(
            kinds=['channel', 'waveform'],
            attributes=dict(quantity=self.quantity[0]))]

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

                if in_codes not in codes_to_traces:
                    continue

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
                        logger.warning(
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

                    except (TraceTooShort, NoData, InvalidResponseError):
                        # print('trace too short: %s' % tr.summary)
                        pass

        return traces_out


class Shift(Operator):
    name = String.T(default='shift')
    codes_projection = basic_codes_projection_t(
        '{i.network}.{i.station}.{i.location}.{i.channel}.{i.extra}S')
    delay = Duration.T()


class Transform(Operator):
    name = String.T(default='trans')
    codes_projection = basic_codes_projection_t(
        '{i.network}.{i.station}.{i.location}'
        '.{i.channel_no_component}{o.component}'
        '.{i.extra}T{o.system}')

    @property
    def kind_provides(self):
        return ('channel', 'waveform')

    @property
    def kind_requires(self):
        return ('channel', 'waveform')

    def get_outlets(self):
        return [
            Outlet(
                kinds=['channel', 'waveform'],
                attributes=dict(
                    system=self.components.lower(),
                    component=c))
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
    name = String.T(default='enz')
    components = 'ENZ'

    def project(self, sensor, trs_sensor):
        return sensor.project_to_enz(trs_sensor)

    def get_orientation(self, sensor, component):
        return {
            'E': (90., 0.),
            'N': (0., 0.),
            'Z': (0., -90.)}[component]


class ToTRZ(Transform):
    name = String.T(default='trz')
    components = 'TRZ'
    origin = Location.T(optional=True)
    azimuth = Float.T(optional=True)

    def project(self, sensor, trs_sensor):
        return sensor.project_to_trz(
            source=self.origin,
            traces=trs_sensor,
            azimuth=self.azimuth)

    def get_orientation(self, sensor, component):
        if self.azimuth is not None:
            azimuth = self.azimuth
        else:
            azimuth = self.origin.azibazi_to(sensor)[1] + 180

        return {
            'T': ((azimuth + 90 + 180.) % 360. - 180., 0.),
            'R': ((azimuth + 180.) % 360. - 180., 0.),
            'Z': (0, -90.)}[component]


class ToLQT(Transform):
    name = String.T(default='lqt')
    components = 'LQT'
    origin = Location.T(optional=True)
    earthmodel = Earthmodel1D.T(optional=True)
    phases = List.T(String.T())
    distance = Float.T(optional=True)
    source_depth = Float.T(optional=True)
    azimuth = Float.T(optional=True)
    incidence = Float.T(optional=True)

    def project(self, sensor, trs_sensor):

        return sensor.project_to_lqt(
            source=self.origin,
            traces=trs_sensor,
            earthmodel=self.earthmodel,
            phases=self.phases,
            distance=self.distance,
            source_depth=None,
            azimuth=None,
            incidence=None)

    def get_orientation(self, sensor, component):
        from pyrocko import cake

        if self.azimuth is None:
            azimuth = self.origin.azibazi_to(sensor)[1] + 180.
        else:
            azimuth = self.azimuth

        if self.incidence is None:
            incidence = sensor.incidence_angle(
                source=self.origin,
                earthmodel=self.earthmodel,
                phases=[cake.PhaseDef(name) for name in self.phases],
                distance=self.distance,
                source_depth=self.source_depth)
        else:
            incidence = self.incidence

        return {
            'L': ((azimuth + 180.) % 360. - 180., incidence),
            'Q': ((azimuth + 180.) % 360. - 180., 90.-incidence),
            'T': ((azimuth + 90. + 180.) % 360. - 180., 0.)}[component]


class CarpetSum(Operator):
    name = String.T(default='csum')

    codes_projection = basic_codes_projection_t('.SUM...')
    nan_aware = Bool.T(default=False, help='Treat NaNs as zero.')
    normalize = Bool.T(default=False, help='Calculate average.')

    @property
    def kind_requires(self):
        return ('carpet',)

    @property
    def kind_provides(self):
        return ('carpet',)

    def get_outlets(self):
        return [Outlet(kinds=['carpet'])]

    def process_carpets(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_carpets: dict[CodesNSLCE, Carpet],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        carpets_out = []
        for mapping in mappings:
            carpets = []
            for codes in mapping.in_codes:
                carpets.extend(codes_to_carpets[codes])

            try:
                carpet = sum_carpets(
                    carpets,
                    nan_aware=self.nan_aware,
                    normalize=self.normalize)

                carpet.codes, = mapping.out_codes
                carpets_out.append(carpet)

            except CarpetError as e:
                logger.warn(str(e))

        return carpets_out


class CarpetHOverV(Operator):
    name = String.T(default='chv')

    codes_projection = basic_codes_projection_t(
        '{i.network}.{i.station}.{i.location}.'
        '{i.channel_no_component}.{i.extra}HV')

    @property
    def kind_requires(self):
        return ('carpet',)

    @property
    def kind_provides(self):
        return ('carpet',)

    def get_outlets(self):
        return [Outlet(kinds=['carpet'])]

    def process_carpets(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_carpets: dict[CodesNSLCE, Carpet],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        carpets_out = []
        for mapping in mappings:
            carpets = []
            for codes in mapping.in_codes:
                carpets.extend(codes_to_carpets[codes])

            carpets_h = [
                carpet for carpet in carpets
                if carpet.codes.channel[-1] in 'NE']
            carpets_v = [
                carpet for carpet in carpets
                if carpet.codes.channel[-1] in 'Z']

            try:
                carpet = sum_carpets(
                    carpets_h,
                    nan_aware=False,
                    normalize=True)

                carpet_v_avg = sum_carpets(
                    carpets_v,
                    nan_aware=False,
                    normalize=True)

                carpet.data -= carpet_v_avg.data

                carpet.codes, = mapping.out_codes
                carpets_out.append(carpet)

            except CarpetError as e:
                logger.warn(str(e))

        return carpets_out


__all__ = [
    'CodesConvertible',
    'HasTimeAndCodes',
    'CodesFilterBase',
    'CodesFilter',
    'CodesProjectionBase',
    'CodesProjection',
    'BaseOperator',
    'Operator',
    'Restitution',
    'Shift',
    'ToENZ',
    'ToTRZ',
    'ToLQT',
    'CarpetSum',
    'CarpetHOverV',
]
