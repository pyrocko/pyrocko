# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
On-demand, chainable data-processing operators.

An :py:class:`Operator` (via its base class :py:class:`BaseOperator`)
presents the same query interface as
:py:class:`~pyrocko.squirrel.base.Squirrel` (``get_waveforms``,
``get_channels``, ``get_responses``, ``get_carpets``, ``get_sensors``,
``chopper_waveforms``, ``chopper_carpets``, ``get_coverage``, etc.), but
derives its content lazily, on each call, from one or more *input*
objects, which may themselves be a
:py:class:`~pyrocko.squirrel.base.Squirrel` or another operator. This
makes operators chainable/stackable: connecting a sequence of operators
(typically declared as a
:py:class:`~pyrocko.squirrel.mantra.Mantra`) forms an on-demand
processing pipeline that can be queried exactly like a squirrel, with
each stage transforming what it receives from its input before handing
it to its consumer. No processing happens until content is actually
requested; nothing is precomputed or cached to disk by the operator
itself.

Each operator advertises which content kinds it needs from its input via
:py:attr:`BaseOperator.kind_requires` and which it offers to its consumer
via :py:attr:`BaseOperator.kind_provides`. Which *codes* (channel/sensor
identifiers) it can produce, and from which input codes each is derived,
is tracked in a codes mapping (:py:class:`CodesMapping`), kept up to
date against the input's currently available codes via
:py:meth:`BaseOperator.update_mappings`. For the common case handled by
:py:class:`Operator`, that mapping is driven by a
:py:attr:`~Operator.codes_projection`
(:py:class:`CodesProjection`): input codes are grouped by a template key
and projected to new output codes via a second template, with
``{i.*}``/``{o.*}`` placeholders referring to the input codes'
attributes and the producing :py:class:`Outlet`'s attributes,
respectively (e.g. :py:class:`Restitution` appends
``R{o.quantity}`` to derive its output channel codes).

The actual content transformation for a given content kind is
implemented by overriding the corresponding ``process_*`` method (e.g.
:py:meth:`BaseOperator.process_waveforms`); :py:class:`BaseOperator`'s
own implementations mostly pass content through unchanged, appropriate
for operators which only rename/reproject codes.

An operator may have more than one input (fan-in), transparently
combined via :py:class:`InputCombinator`; which operators feed which is
normally set up declaratively through a
:py:class:`~pyrocko.squirrel.mantra.Mantra`'s
:py:meth:`~pyrocko.squirrel.mantra.Mantra.setup`, using operators'
``input_names``, rather than by calling
:py:meth:`BaseOperator.add_input` directly.
'''

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
    Object, String, Float, Bool, clone, List, Dict, equal
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
    '''
    Raised on operator setup/wiring errors, e.g. incompatible input (see
    :py:meth:`BaseOperator.add_input`) or an unsupported multi-squirrel
    setup (see :py:meth:`InputCombinator.get_squirrel`).
    '''


def scodes(codes):
    '''
    Get compact textual summary of a group of codes, for display purposes.

    :param codes:
        Codes to be summarized.
    :type codes:
        iterable of :py:class:`~pyrocko.squirrel.model.Codes` objects, all
        of the same class

    :returns:
        If all but one code element (e.g. network, station, ...) agree
        across ``codes``, a single dot-separated string with the
        differing element rendered as a comma-separated ``{...}`` group
        (e.g. ``'NET.STA..{HHE,HHN,HHZ}.'``). Otherwise, the codes listed
        out individually, comma-separated.
    :rtype:
        str
    '''
    css = list(zip(*codes))
    if sum(not all(c == cs[0] for c in cs) for cs in css) == 1:
        return '.'.join(
            cs[0] if all(c == cs[0] for c in cs) else '{%s}' % ','.join(cs)
            for cs in css)
    else:
        return ', '.join(str(c) for c in codes)


def time_min_max(codes_to_traces):
    '''
    Get overall time span of a codes-to-traces mapping.

    :param codes_to_traces:
        Traces to inspect, keyed by codes (only the values are used).
    :type codes_to_traces:
        dict of :py:class:`list` of :py:class:`~pyrocko.trace.Trace`

    :returns:
        ``(tmin, tmax)``, the earliest start and latest end time across
        all traces.
    :rtype:
        :py:class:`tuple`
    '''
    trs = lchain(codes_to_traces.values())
    tmin_trs = min(tr.tmin for tr in trs)
    tmax_trs = max(tr.tmax for tr in trs)
    return tmin_trs, tmax_trs


def by_codes(xs: HasCodes) -> dict[CodesNSLCE, tList[HasCodes]]:
    '''
    Group objects with a ``codes`` attribute by their codes.

    :param xs:
        Objects to be grouped, each having a ``codes`` attribute (e.g.
        :py:class:`~pyrocko.trace.Trace`,
        :py:class:`~pyrocko.squirrel.model.Channel`,
        :py:class:`~pyrocko.squirrel.model.Response`).
    :type xs:
        iterable

    :rtype:
        dict of :py:class:`~pyrocko.squirrel.model.CodesNSLCE` to
        :py:class:`list`
    '''
    by_codes = defaultdict(list)
    for x in xs:
        by_codes[x.codes].append(x)

    return by_codes


def by_codes_unique(xs: HasCodes) -> dict[CodesNSLCE, HasCodes]:
    '''
    Like :py:func:`by_codes`, but keeping only codes with exactly one
    matching object, unwrapped from their single-element list.

    :param xs:
        Objects to be grouped, each having a ``codes`` attribute.
    :type xs:
        iterable

    :rtype:
        dict of :py:class:`~pyrocko.squirrel.model.CodesNSLCE` to object
    '''
    return dict(
        (k, xs_group[0])
        for (k, xs_group) in by_codes(xs).items()
        if len(xs_group) == 1)


class CodesFilterBase(Object):
    '''
    Base class for codes filters, used to select which of an operator's
    input codes it should act on (see
    :py:meth:`BaseOperator.update_mappings`).
    '''

    __eq__ = equal

    def filter(self, it: Sequence[CodesNSLCE]) -> List[CodesNSLCE]:
        '''
        Filter a sequence of codes.

        :param it:
            Codes to be filtered.
        :type it:
            iterable of :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :returns:
            The base implementation returns all given codes unfiltered;
            subclasses restrict this.
        :rtype:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`
        '''
        return list(it)


class CodesFilter(CodesFilterBase):
    '''
    Filter by codes patterns.
    '''
    include = List.T(
        CodesNSLCE.T(),
        optional=True,
        help='If given, only codes matching one of these patterns pass '
             'the filter.')
    exclude = List.T(
        CodesNSLCE.T(),
        optional=True,
        help='If given, codes matching one of these patterns are removed '
             'from the filter result, even if they match ``include``.')

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
        '''
        Check if a single codes object passes the filter.

        :param codes:
            Codes to be checked.
        :type codes:
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :rtype:
            bool
        '''
        return (self._matcher is None or self._matcher.match(codes)) \
            and (self.exclude is None
                 or not self._matcher_exclude.match(codes))

    def filter(self, it: Sequence[CodesNSLCE]) -> List[CodesNSLCE]:
        '''
        Filter a sequence of codes by :py:attr:`include`/
        :py:attr:`exclude`.

        :param it:
            Codes to be filtered.
        :type it:
            iterable of :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :rtype:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`
        '''
        if self._matcher is None:
            return list(it)
        else:
            return list(self._matcher.filter(it))


class CodesMapping:
    '''
    Records which output codes an operator derives from which input
    codes, for a single output group (internal use, see
    :py:meth:`BaseOperator.iter_mappings`).

    *Attributes:*

    .. py:attribute:: in_codes_set

        Working set of input codes belonging to this mapping, kept in
        sync incrementally as the input's available codes change (see
        :py:meth:`Operator._update_mappings_specific`).

    .. py:attribute:: in_codes

        Input codes belonging to this mapping, as a sorted tuple
        (refreshed from :py:attr:`in_codes_set` whenever it changes).

    .. py:attribute:: out_codes

        Output codes derived from :py:attr:`in_codes`, one per
        :py:class:`Outlet` returned by
        :py:meth:`BaseOperator.get_outlets`.

    .. py:attribute:: group_key

        Key under which this mapping is grouped (e.g. the input codes'
        NSLC-minus-component identifier), as computed by the operator's
        grouping logic.
    '''

    __slots__ = ['in_codes_set', 'in_codes', 'out_codes', 'group_key']

    def __init__(self):
        self.in_codes_set = set()
        self.in_codes = ()
        self.out_codes = ()
        self.group_key = ''

    def describe(self):
        '''
        Get one-line summary, as ``'<n_out> <- <n_in>'``.

        :rtype:
            str
        '''
        return '%i <- %i' % (len(self.out_codes), len(self.in_codes))


def new_operator_id():
    '''
    Generate a fresh unique id for an operator instance (see
    :py:meth:`BaseOperator.reset`).

    :rtype:
        :py:class:`uuid.UUID`
    '''
    return uuid.uuid4()


class Outlet(Object):
    '''
    Describes one distinct output "channel" of an operator, e.g. one
    component of a 3-component coordinate transform.

    Used by :py:meth:`BaseOperator.get_outlets` and consumed by
    :py:meth:`CodesProjection.project` to fill the ``{o.*}`` placeholders
    when deriving output codes.
    '''

    kinds = List.T(
        String.T(),
        help='Content kinds produced through this outlet, e.g. '
             "``['channel', 'waveform']``.")
    attributes = Dict.T(
        String.T(), String.T(),
        help='Named values substituted for ``{o.<name>}`` placeholders '
             'in the operator\'s codes-projection template, e.g. '
             "``{'component': 'Z'}``.")


class InputCombinator:
    '''
    Transparently fans out queries to several inputs and aggregates the
    results, giving a combined input the same interface as a single
    operator/squirrel input (internal use, see
    :py:meth:`BaseOperator.add_input`).
    '''

    def __init__(self, inputs):
        '''
        :param inputs:
            Inputs to be combined.
        :type inputs:
            iterable of operators or
            :py:class:`~pyrocko.squirrel.base.Squirrel`
        '''
        self._inputs = []
        self.reset()
        for input in inputs:
            self.add_input(input)

    def add_input(self, input):
        '''
        Add an input to the combination.

        :param input:
            Input to add.
        :type input:
            operator or :py:class:`~pyrocko.squirrel.base.Squirrel`

        :raises OperatorError:
            If ``input`` is already among the combined inputs.
        '''
        if input in self._inputs:
            raise OperatorError(
                f'Input "{input.name}" already among inputs.')

        self._inputs.append(input)
        self.reset()

    def reset(self):
        '''
        Reset cached mapping-invalidation state (see
        :py:meth:`update_mappings`).
        '''
        self._input_mapping_counters = None
        self._mapping_counter = 0

    def update_mappings(self):
        '''
        Propagate a mapping update to all combined inputs and get a
        counter reflecting the combined update state.

        :returns:
            A counter, incremented whenever any combined input's own
            mapping-update counter has changed since the last call (see
            :py:meth:`BaseOperator.update_mappings`).
        :rtype:
            int
        '''
        if self._input_mapping_counters is None:
            self._input_mapping_counters = [None] * len(self._inputs)

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

        '''
        Get combined time span across all inputs.

        :param kinds:
            Content kinds to consider.
        :type kinds:
            :py:class:`list` of :py:class:`str`

        :param dummy_limits:
            Passed through to each input's own ``get_time_span``.
        :type dummy_limits:
            bool

        :returns:
            ``(tmin, tmax)``, the earliest start and latest end time
            across all inputs, or ``(None, None)`` if no input reports a
            defined time span.
        :rtype:
            :py:class:`tuple`
        '''

        tmins, tmaxs = zip(*(
            input.get_time_span(kinds, dummy_limits=dummy_limits)
            for input in self._inputs))

        tmins = [tmin for tmin in tmins if tmin is not None]
        tmaxs = [tmax for tmax in tmaxs if tmax is not None]

        if not tmins or not tmaxs:
            return (None, None)

        return min(tmins), max(tmaxs)

    # The following methods all fan a query out to every combined input
    # and merge the results: the get_{in_,}codes family deduplicates and
    # sorts into a combined set/list (iter_* yielding an iterator over
    # the same), while get_carpets/get_channels/get_events/get_waveforms/
    # get_responses/get_rich_coverage/get_sensors simply concatenate each
    # input's results, keyword arguments passed through unchanged.

    def iter_in_codes(self, *args, **kwargs):
        '''
        Iterate over combined, deduplicated, sorted input codes across
        all inputs. See :py:meth:`BaseOperator.iter_in_codes`.
        '''
        return iter(
            sorted(set(self._aggregate('get_in_codes', *args, **kwargs))))

    def iter_codes(self, *args, **kwargs):
        '''
        Iterate over combined, deduplicated, sorted output codes across
        all inputs. See :py:meth:`BaseOperator.iter_codes`.
        '''
        return iter(
            sorted(set(self._aggregate('get_codes', *args, **kwargs))))

    def get_in_codes(self, *args, **kwargs):
        '''
        Get combined, deduplicated, sorted input codes across all inputs.
        See :py:meth:`BaseOperator.get_in_codes`.
        '''
        return sorted(set(self._aggregate('get_in_codes', *args, **kwargs)))

    def get_codes(self, *args, **kwargs):
        '''
        Get combined, deduplicated, sorted output codes across all
        inputs. See :py:meth:`BaseOperator.get_codes`.
        '''
        return sorted(set(self._aggregate('get_codes', *args, **kwargs)))

    def get_carpets(self, *args, **kwargs):
        '''
        Get concatenated carpets across all inputs. See
        :py:meth:`BaseOperator.get_carpets`.
        '''
        return self._aggregate('get_carpets', *args, **kwargs)

    def get_channels(self, *args, **kwargs):
        '''
        Get concatenated channels across all inputs. See
        :py:meth:`BaseOperator.get_channels`.
        '''
        return self._aggregate('get_channels', *args, **kwargs)

    def get_events(self, *args, **kwargs):
        '''
        Get concatenated events across all inputs. See
        :py:meth:`BaseOperator.get_events`.
        '''
        return self._aggregate('get_events', *args, **kwargs)

    def get_waveforms(self, *args, **kwargs):
        '''
        Get concatenated waveforms across all inputs. See
        :py:meth:`BaseOperator.get_waveforms`.
        '''
        return self._aggregate('get_waveforms', *args, **kwargs)

    def get_responses(self, *args, **kwargs):
        '''
        Get concatenated responses across all inputs. See
        :py:meth:`BaseOperator.get_responses`.
        '''
        return self._aggregate('get_responses', *args, **kwargs)

    def get_rich_coverage(self, *args, **kwargs):
        '''
        Get concatenated rich coverage across all inputs. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_rich_coverage`.
        '''
        return self._aggregate('get_rich_coverage', *args, **kwargs)

    def get_sensors(self, *args, **kwargs):
        '''
        Get concatenated sensors across all inputs. See
        :py:meth:`BaseOperator.get_sensors`.
        '''
        return self._aggregate('get_sensors', *args, **kwargs)

    def get_squirrel(self, *args, **kwargs):
        '''
        Get the common root squirrel of all combined inputs.

        :rtype:
            :py:class:`~pyrocko.squirrel.base.Squirrel`

        :raises OperatorError:
            If the combined inputs do not share a single common root
            squirrel.
        '''
        from ..base import Squirrel
        squirrels = [
            input if isinstance(input, Squirrel) else input.get_squirrel()
            for input in self._inputs]

        if not all(squirrels[0] is squirrel for squirrel in squirrels):
            raise OperatorError(
                'Currently, only a single root Squirrel is supported.')

        return squirrels[0]

    def advance_accessor(self, *args, **kwargs):
        '''
        Advance the memory-cache accessor on all combined inputs. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.advance_accessor`.
        '''
        for input in self._inputs:
            input.advance_accessor(*args, **kwargs)


class BaseOperator(Object):
    '''
    Base class for on-demand data-processing operators.

    Provides an operator's generic plumbing (input wiring, codes-mapping
    bookkeeping, the squirrel-like query methods dispatching through
    ``process_*``); see the module docstring for the overall concept.
    Most concrete operators derive from :py:class:`Operator` instead,
    which adds the common filter/group/translate ``codes_projection``
    machinery; :py:class:`BaseOperator` is for operators needing full
    control over their codes mapping.
    '''

    name = String.T(
        default='base_op',
        help='Name of this operator, used to refer to it from other '
             'operators\' ``input_names`` and as a namespacing element '
             'in derived data codes.')
    input_names = List.T(
        String.T(),
        optional=True,
        help='Names of operators (or ``\'squirrel\'``) to take as input. '
             'If not given, this operator implicitly takes the previous '
             'operator (or the squirrel, for the first one) in its '
             ':py:class:`~pyrocko.squirrel.mantra.Mantra`\'s operator '
             'list; see '
             ':py:meth:`~pyrocko.squirrel.mantra.Mantra._plan_connections`.')

    def post_init(self):
        '''
        Guts post-initialization hook, resetting runtime state (see
        :py:meth:`reset`).
        '''
        self.reset()
        if self.name is None:
            self.name = self.__class__.__name__

    def reset(self):
        '''
        Reset all runtime state (input wiring, codes mapping, mantra
        association) to a freshly-constructed state.
        '''
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
        '''
        Content kinds this operator offers to its consumer.

        :rtype:
            :py:class:`tuple` of :py:class:`str`
        '''
        return ('channel', 'response', 'waveform')

    @property
    def kind_requires(self):
        '''
        Content kinds this operator needs from its input.

        :rtype:
            :py:class:`tuple` of :py:class:`str`
        '''
        return ()

    def add_input(self, input: Operator | Squirrel) -> None:
        '''
        Add an input to this operator.

        :param input:
            Input to add. Multiple inputs are combined transparently
            (see :py:class:`InputCombinator`); normally, inputs are
            wired up declaratively via a
            :py:class:`~pyrocko.squirrel.mantra.Mantra` rather than by
            calling this directly.
        :type input:
            operator or :py:class:`~pyrocko.squirrel.base.Squirrel`

        :raises Exception:
            If ``input`` does not provide a content kind listed in
            :py:attr:`kind_requires`.
        '''
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
        '''
        Get this operator's input (see :py:meth:`add_input`).

        :rtype:
            operator, :py:class:`InputCombinator` or
            :py:class:`~pyrocko.squirrel.base.Squirrel`, or ``None`` if
            no input has been added yet
        '''
        return self._input

    def set_mantra(self, mantra):
        '''
        Associate this operator with the mantra it belongs to (internal
        use, called from
        :py:meth:`~pyrocko.squirrel.mantra.Mantra.setup`).

        :param mantra:
            Owning mantra.
        :type mantra:
            :py:class:`~pyrocko.squirrel.mantra.Mantra`
        '''
        self._mantra = mantra

    @property
    def mantra_name(self):
        '''
        Name of the owning :py:class:`~pyrocko.squirrel.mantra.Mantra`
        (see :py:meth:`set_mantra`), or ``''`` if not (yet) associated
        with one.
        '''
        return self._mantra.name if self._mantra is not None else ''

    def describe(self) -> str:
        '''
        Get textual description of this operator, its outlets and its
        current codes mapping.

        :rtype:
            str
        '''
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

        '''
        Iterate over this operator's current codes mappings, refreshing
        them first (see :py:meth:`update_mappings`).

        :param codes:
            If given, restrict to mappings with at least one matching
            output code.
        :type codes:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :yields:
            :py:class:`CodesMapping` objects.
        '''

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

        '''
        Get this operator's current codes mappings, see
        :py:meth:`iter_mappings`.
        '''

        return list(self.iter_mappings(codes))

    def get_mappings_and_matching_codes(
                self,
                codes: tList[CodesNSLCE] | None = None
                ) -> (tList[CodesMapping], tSet[CodesNSLCE]):

        '''
        Get codes mappings together with the subset of their output codes
        matching a query, refreshing mappings first (see
        :py:meth:`update_mappings`).

        :param codes:
            If given, restrict to mappings with at least one matching
            output code, and report only the matching output codes.
        :type codes:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :returns:
            ``(mappings, codes_out_match)`` if ``codes`` is given (an
            empty pair if no mapping matches), else ``(all mappings,
            None)``.
        :rtype:
            :py:class:`tuple` of (:py:class:`list` of
            :py:class:`CodesMapping`, :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE` or ``None``)
        '''

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
        '''
        Iterate over this operator's output codes, refreshing mappings
        first (see :py:meth:`update_mappings`).

        :yields:
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE` objects.
        '''

        self.update_mappings()

        for mapping in self._mappings.values():
            yield from mapping.out_codes

    def get_codes(self, kind: str = None) -> tList[CodesNSLCE]:
        '''
        Get this operator's output codes for a given content kind.

        :param kind:
            Content kind to query. With ``None``, the codes of all provided
            kinds are returned. A kind not listed in :py:attr:`kind_provides`
            yields an empty result rather than an error.
        :type kind:
            str

        :rtype:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`
        '''
        if kind is not None and kind not in self.kind_provides:
            return []

        return list(self.iter_codes())

    def iter_in_codes(
                self,
                mappings: tList[CodesMapping] | None = None
            ) -> Generator[CodesNSLCE]:

        '''
        Iterate over this operator's input codes, refreshing mappings
        first (see :py:meth:`update_mappings`).

        :param mappings:
            If given, restrict to these mappings' input codes instead of
            all current mappings.
        :type mappings:
            :py:class:`list` of :py:class:`CodesMapping`

        :yields:
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE` objects.
        '''

        self.update_mappings()

        if mappings is None:
            mappings = self._mappings.values()

        for mapping in mappings:
            yield from mapping.in_codes

    def get_in_codes(
                self,
                mappings: tList[CodesMapping] | None = None
            ) -> tList[CodesNSLCE]:

        '''
        Get this operator's input codes, sorted, see
        :py:meth:`iter_in_codes`.
        '''

        return sorted(self.iter_in_codes(mappings))

    def update_mappings(self) -> None:
        '''
        Refresh this operator's codes mapping against its input's
        currently available codes, if needed.

        Cheap to call repeatedly: the input's own update counter (see
        :py:meth:`~pyrocko.squirrel.base.Squirrel.update_mappings`-alike
        methods on inputs) is checked first, and no work is done unless
        it has changed since the last call. When it has, the set of
        available input codes (intersected across all
        :py:attr:`kind_requires` kinds, or unfiltered by kind if
        :py:attr:`kind_requires` is empty) is diffed against the
        previous set, and added/removed codes are passed to
        :py:meth:`_update_mappings_specific` (implemented by subclasses,
        e.g. :py:meth:`Operator._update_mappings_specific`) to update
        :py:attr:`_mappings`.

        :returns:
            A counter, incremented whenever the mapping actually changed.
        :rtype:
            int
        '''

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

    # The get_in_{channels,responses,waveforms,carpets,coverages} family
    # queries the operator's input for raw content matching given input
    # codes and time span, and groups the result by codes (see
    # by_codes()/by_codes_unique()) for consumption by the matching
    # process_*() method.

    def get_in_channels(
                self,
                in_codes: tList[CodesNSLCE],
                tmin: TimeFloat,
                tmax: TimeFloat,
            ) -> dict[CodesNSLCE, Channel]:

        '''
        Get input channels, grouped by codes. See :py:meth:`by_codes
        <pyrocko.squirrel.operators.base.by_codes>` and
        :py:meth:`process_channels`.
        '''

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

        '''
        Get input responses, grouped by codes. See
        :py:meth:`process_responses`.
        '''

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

        '''
        Get input waveforms, grouped by codes. See
        :py:meth:`process_waveforms`.
        '''

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

        '''
        Get input carpets, grouped by codes. See
        :py:meth:`process_carpets`.
        '''

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

        '''
        Get input coverage, grouped by codes, keeping at most one
        coverage entry per codes (see :py:meth:`by_codes_unique
        <pyrocko.squirrel.operators.base.by_codes_unique>`). See
        :py:meth:`process_coverage`.
        '''

        coverages = self._input.get_coverage(
            kind, codes=in_codes, tmin=tmin, tmax=tmax, **kwargs)

        return by_codes_unique(coverages)

    def get_rich_coverage(
            self, tmin=None, tmax=None, codes=None, limit=None):

        '''
        Get combined coverage information for channel/response/waveform
        content kinds, derived through this operator. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_rich_coverage`.
        '''

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

        return [
            make_rich_coverage(kinds, coverages)
            for _, coverages in coverages_by_codes.items()]

    # The get_{channels,sensors,responses,waveforms,carpets}() methods
    # below all follow the same shape: resolve the query's selection
    # args, look up which input codes map to the requested output codes
    # (get_mappings[_and_matching_codes]()), fetch the matching raw
    # input content (get_in_*()), run it through the corresponding
    # process_*() method to derive the output content, then filter down
    # to exactly the requested output codes. Together, this makes an
    # operator's query interface behave like
    # :py:class:`~pyrocko.squirrel.base.Squirrel`'s, transparently
    # inserting the operator's processing step.

    def get_channels(
                self,
                obj: HasTimeAndCodes = None,
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
                time: TimeFloat = None,
                codes: CodesConvertible = None,
            ) -> tList[Channel]:

        '''
        Get channels derived through this operator. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_channels`.
        '''

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

        '''
        Get sensors derived through this operator, by grouping
        :py:meth:`get_channels`' result (see :py:meth:`process_sensors`).
        See also
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_sensors`-alike
        querying.
        '''

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

        '''
        Get instrument responses derived through this operator. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_responses`.
        '''

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

        '''
        Get waveforms derived through this operator, padded by
        :py:meth:`get_time_padding` on input before processing (e.g. to
        give a filter its transient time). See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_waveforms`.
        '''

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

        '''
        Get carpets derived through this operator, padded by
        :py:meth:`get_time_padding` on input before processing. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_carpets`.
        '''

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

        '''
        Iterate over carpets derived through this operator, in
        consecutive time windows. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_carpets` and
        :py:func:`pyrocko.util.iter_windows` for windowing parameters.

        :yields:
            :py:class:`~pyrocko.squirrel.base.Batch` objects, with a
            ``carpets`` attribute holding each window's carpets (see
            :py:meth:`get_carpets`).
        '''

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

        '''
        Get events. Not currently implemented by any operator; always
        returns an empty list, regardless of input.
        '''

        return []

    def get_squirrel(self):
        '''
        Get the root :py:class:`~pyrocko.squirrel.base.Squirrel` this
        operator ultimately derives its content from, following the
        input chain.

        :rtype:
            :py:class:`~pyrocko.squirrel.base.Squirrel`
        '''
        from ..base import Squirrel
        if self._input is None or isinstance(self._input, Squirrel):
            return self._input
        else:
            return self._input.get_squirrel()

    def advance_accessor(self, accessor_id='default', cache_id=None):
        '''
        Advance the memory-cache accessor on the input. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.advance_accessor`.
        '''
        self._input.advance_accessor(accessor_id, cache_id)

    def chopper_waveforms(
            self, obj=None, tmin=None, tmax=None, time=None, codes=None,
            codes_exclude=None, sample_rate_min=None, sample_rate_max=None,
            tinc=None, tpad=0.,
            want_incomplete=True, snap_window=False,
            degap=True, maxgap=5, maxlap=None,
            snap=None, include_last=False, load_data=True,
            accessor_id=None, clear_accessor=True,
            group_by=None, channel_priorities=None):

        '''
        Iterate over waveforms derived through this operator, in
        consecutive time windows.

        :param tinc:
            Time window length [s]. If not given, a single window
            covering the whole queried time span is used.
        :type tinc:
            float

        :param tpad:
            Padding time [s] added symmetrically to each window before
            processing (see :py:meth:`get_waveforms`) and reported back
            via :py:attr:`~pyrocko.squirrel.base.Batch.tpad`.
        :type tpad:
            float

        :param snap_window:
            If ``True``, snap window start/end to multiples of ``tinc``.
        :type snap_window:
            bool

        :param group_by:
            If given, a codes-projection group-by template (see
            :py:attr:`CodesProjection.group_by`) to additionally split
            each time window into separate batches per matching group of
            output codes (e.g. to chop each sensor separately). By
            default, all matching codes are chopped together into a
            single group per window.
        :type group_by:
            str

        :param accessor_id:
            Name of consumer on whose behalf data is accessed (see
            :py:meth:`~pyrocko.squirrel.base.Squirrel.get_waveforms`). By
            default, a name unique to this operator instance and chopper
            invocation is generated.
        :type accessor_id:
            str

        :param clear_accessor:
            If ``True`` (the default), release cached waveform data held
            for ``accessor_id`` once the chopper is exhausted or aborted.
        :type clear_accessor:
            bool

        :param channel_priorities:
            See
            :py:meth:`~pyrocko.squirrel.base.Squirrel.get_waveforms`'s
            parameter of the same name.

        See :py:meth:`get_waveforms` for the remaining parameters
        (``obj``, ``tmin``, ``tmax``, ``time``, ``codes``,
        ``codes_exclude``, ``sample_rate_min``, ``sample_rate_max``,
        ``want_incomplete``, ``degap``, ``maxgap``, ``maxlap``, ``snap``,
        ``include_last``, ``load_data``), applied per window.

        :yields:
            :py:class:`~pyrocko.squirrel.base.Batch` objects, one per
            time window (times ``ngroups``, if ``group_by`` is given).
        '''

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

                operator.add_input(self)

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

        '''
        Get coverage information derived through this operator. See
        :py:meth:`~pyrocko.squirrel.base.Squirrel.get_coverage`.
        '''

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

    # The process_{channels,responses,waveforms,carpets,coverage} methods
    # are where subclasses implement their actual content transformation
    # (e.g. Restitution.process_waveforms(), Transform.process_channels()
    # /process_waveforms()); each is called once per get_*() query, with
    # already-fetched, by-codes-grouped input content and the relevant
    # CodesMapping objects, and returns the corresponding output content
    # with codes set to each mapping's out_codes. The defaults below
    # implement that for operators which only need to filter/rename, not
    # transform, content.

    def process_channels(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_channels: dict[CodesNSLCE, tList[Channel]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Channel]:

        '''
        Derive output channels from input channels, cloning and
        reprojecting codes according to ``mappings`` (default
        implementation, suitable for operators which do not need to
        modify channel objects themselves).

        :param mappings:
            Codes mappings to derive output channels for (see
            :py:meth:`get_mappings`/:py:meth:`get_mappings_and_matching_\
codes`).
        :type mappings:
            :py:class:`list` of :py:class:`CodesMapping`

        :param in_codes:
            Unused in the default implementation (present for API
            consistency and for use by overriding implementations).
        :type in_codes:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :param codes_to_channels:
            Input channels, keyed by input codes (see
            :py:meth:`get_in_channels`).
        :type codes_to_channels:
            dict

        :param tmin:
            Unused in the default implementation.

        :param tmax:
            Unused in the default implementation.

        :rtype:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.Channel`
        '''

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

        '''
        Group channels into sensors (default implementation, see
        :py:meth:`~pyrocko.squirrel.model.Sensor.from_channels`).

        :param channels:
            Output channels to be grouped, as returned by
            :py:meth:`process_channels`.
        :type channels:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.Channel`

        :rtype:
            :py:class:`list` of :py:class:`~pyrocko.squirrel.model.Sensor`
        '''

        return Sensor.from_channels(channels)

    def process_responses(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_responses: dict[CodesNSLCE, tList[Response]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Response]:

        '''
        Derive output responses from input responses, cloning and
        reprojecting codes according to ``mappings``. Parameters as for
        :py:meth:`process_channels`.

        :rtype:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.Response`
        '''

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
                codes_to_traces: dict[CodesNSLCE, tList[Trace]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        '''
        Derive output waveforms from input waveforms, copying and
        reprojecting codes according to ``mappings`` (default
        implementation, suitable for operators which do not transform the
        samples; see :py:meth:`Restitution.process_waveforms` for an
        example which does). The copies share the sample data with the
        input traces.

        Parameters as for :py:meth:`process_channels`, with
        ``codes_to_traces`` in place of ``codes_to_channels``.

        :rtype:
            :py:class:`list` of :py:class:`~pyrocko.trace.Trace`
        '''

        traces = []
        for mapping in mappings:
            for in_codes, out_codes in zip(
                    mapping.in_codes, mapping.out_codes):

                for trace_in in codes_to_traces.get(in_codes, []):
                    trace = trace_in.copy(data=False)
                    trace.set_codes(*out_codes)
                    traces.append(trace)

        return traces

    def process_carpets(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_carpets: dict[CodesNSLCE, tList[Carpet]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Carpet]:

        '''
        Derive output carpets from input carpets, copying and reprojecting
        codes according to ``mappings`` (default implementation, suitable
        for operators which do not transform the data; subclasses producing
        new carpets override this, e.g. :py:meth:`CarpetSum.process_carpets`).
        The copies share the data with the input carpets.

        Parameters as for :py:meth:`process_channels`, with
        ``codes_to_carpets`` in place of ``codes_to_channels``.

        :rtype:
            :py:class:`list` of :py:class:`~pyrocko.carpet.Carpet`
        '''

        carpets = []
        for mapping in mappings:
            for in_codes, out_codes in zip(
                    mapping.in_codes, mapping.out_codes):

                for carpet_in in codes_to_carpets.get(in_codes, []):
                    carpet = carpet_in.copy(data='reference')
                    carpet.codes = out_codes
                    carpets.append(carpet)

        return carpets

    def process_coverage(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_coverage: dict[CodesNSLCE, Coverage],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        '''
        Derive output coverage from input coverage, by joining
        (:py:func:`~pyrocko.squirrel.model.join_coverages`) each
        mapping's input coverages, padded by :py:meth:`get_time_padding`,
        and re-tagging the result with each of the mapping's
        ``out_codes``.

        Parameters as for :py:meth:`process_channels`, with
        ``codes_to_coverage`` in place of ``codes_to_channels``.

        :rtype:
            :py:class:`list` of
            :py:class:`~pyrocko.squirrel.model.Coverage`
        '''

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
        '''
        Get extra time span [s] this operator needs from its input on
        each side of a query's time span (e.g. filter transient time),
        used by :py:meth:`get_waveforms`/:py:meth:`get_carpets`/
        :py:meth:`get_coverage`/:py:meth:`get_time_span` to pad input
        queries and shrink the reported available time span accordingly.
        Default: ``0.0``; overridden e.g. by
        :py:meth:`Restitution.get_time_padding`.
        '''
        return 0.0

    def get_time_span(
            self,
            kinds,
            dummy_limits=True) -> tTuple[TimeFloat, TimeFloat]:

        '''
        Get time span available through this operator, shrunk on both
        ends by :py:meth:`get_time_padding` relative to the input's own
        time span.

        :param kinds:
            Content kinds to consider.
        :type kinds:
            :py:class:`list` of :py:class:`str`

        :param dummy_limits:
            Passed through to the input's own ``get_time_span``.
        :type dummy_limits:
            bool

        :returns:
            ``(tmin, tmax)``, or ``(None, None)`` if the input reports an
            undefined time span.
        :rtype:
            :py:class:`tuple`
        '''

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
    '''
    Base class for codes projections: combined filter, grouping and
    codes-derivation strategies used by :py:class:`Operator` (see
    :py:attr:`Operator.codes_projection`).
    '''

    def group_key(self, codes):
        '''
        Get the key under which a given input codes should be grouped
        with others sharing the same key (e.g. to combine a sensor's
        component channels into one mapping).

        :param codes:
            Input codes to compute the group key for.
        :type codes:
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`
        '''
        raise NotImplementedError()

    def project(self, operator, codes, outlets):
        '''
        Derive output codes for a group of input codes.

        :param operator:
            Operator this projection belongs to (used e.g. to fill
            ``{o.name}``/``{o.mantra}`` template placeholders), or
            ``None``.
        :type operator:
            :py:class:`Operator` or ``None``

        :param codes:
            Input codes sharing a common :py:meth:`group_key`.
        :type codes:
            iterable of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :param outlets:
            The operator's outlets (see
            :py:meth:`BaseOperator.get_outlets`), one output code is
            derived per outlet.
        :type outlets:
            :py:class:`list` of :py:class:`Outlet`
        '''
        raise NotImplementedError()


class CodesProjection(CodesProjectionBase):
    '''
    Template-string-based codes projection.

    Groups and derives codes using Python ``str.format``-style templates
    with ``{i.<attr>}``/``{o.<attr>}`` placeholders, referring to the
    input codes' attributes (``network``, ``station``, ``location``,
    ``channel``, ``extra``, and the extra attributes defined on
    :py:class:`~pyrocko.squirrel.model.CodesNSLCE`, e.g.
    ``channel_no_component``) and to the producing
    :py:class:`Outlet`'s :py:attr:`~Outlet.attributes` plus the
    operator's ``name``/``mantra`` (unset ``{o.*}`` placeholders resolve
    to the empty string when no outlet/operator context is given, e.g.
    when used standalone for :py:attr:`group_by`).
    '''

    template = String.T(
        default='{i.network}.{i.station}.{i.location}.{i.channel}.{i.extra}',
        help='Template used to derive output codes from a group of '
             'input codes and an outlet, e.g. '
             '``\'{i.network}.{i.station}.{i.location}.{i.channel}\'``.')

    group_by = String.T(
        optional=True,
        help='Template used to compute the grouping key for input codes '
             '(see :py:meth:`group_key`). Defaults to ``template`` '
             'itself, appropriate when the projection does not '
             'aggregate multiple input codes (e.g. no ``{o.*}`` '
             'placeholders).')

    def __init__(self, template=None, group_by=None, **kwargs):
        '''
        :param template:
            See :py:attr:`template`.
        :type template:
            str

        :param group_by:
            See :py:attr:`group_by`.
        :type group_by:
            str
        '''
        d = {}
        if template is not None:
            d['template'] = template

        if group_by is not None:
            d['group_by'] = group_by

        CodesProjectionBase.__init__(self, **d, **kwargs)

    def group_key(self, codes):
        '''
        Get grouping key for input codes, from :py:attr:`group_by` (or
        :py:attr:`template`, if unset).

        :param codes:
            Input codes to compute the group key for.
        :type codes:
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :rtype:
            str
        '''
        return (self.group_by or self.template).format(
            i=codes, o=empty_strings)

    def _project_single(self, operator, codes, outlet):
        if operator:
            d = dict(name=operator.name, mantra=operator.mantra_name)
        else:
            d = {}
        d.update(outlet.attributes)
        o = util.Anon(**d)
        return CodesNSLCE(self.template.format(i=codes, o=o))

    def project(self, operator, codes_group, outlets):
        '''
        Derive output codes for a group of input codes, from
        :py:attr:`template`, for every input-codes/outlet combination.

        :param operator:
            Operator this projection belongs to, or ``None``.
        :type operator:
            :py:class:`Operator` or ``None``

        :param codes_group:
            Input codes sharing a common :py:meth:`group_key`.
        :type codes_group:
            iterable of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :param outlets:
            The operator's outlets.
        :type outlets:
            :py:class:`list` of :py:class:`Outlet`

        :returns:
            Deduplicated, sorted output codes.
        :rtype:
            :py:class:`tuple` of
            :py:class:`~pyrocko.squirrel.model.CodesNSLCE`
        '''
        return tuple(sorted(set(
            self._project_single(operator, codes, outlet)
            for codes in codes_group
            for outlet in outlets)))


def basic_codes_projection_t(template):
    '''
    Get a Guts field definition for a :py:attr:`Operator.codes_projection`
    attribute, defaulting to a :py:class:`CodesProjection` with a given
    template.

    :param template:
        Default :py:attr:`CodesProjection.template` string.
    :type template:
        str

    :rtype:
        Guts field definition (see :py:class:`CodesProjectionBase.T`)
    '''
    return CodesProjectionBase.T(default=CodesProjection.D(template=template))


class Operator(BaseOperator):
    '''
    Base class for operators with typical filter-group-translate
    behaviour.

    Adds the generic :py:attr:`codes_projection`-driven codes-mapping
    machinery on top of :py:class:`BaseOperator`: input codes are
    filtered, grouped and projected to output codes automatically (see
    :py:class:`CodesProjection`), so concrete subclasses (e.g.
    :py:class:`Restitution`, :py:class:`Transform`) typically only need
    to declare a :py:attr:`codes_projection`, override
    :py:meth:`~BaseOperator.get_outlets` and implement the relevant
    ``process_*`` method(s).
    '''

    name = String.T(
        default='op',
        help='Name of this operator, see '
             ':py:attr:`~BaseOperator.name`.')
    codes_projection = basic_codes_projection_t(
        '{i.network}.{i.station}.{i.location}.{i.channel}.{i.extra}')

    def get_outlets(self):
        '''
        Get this operator's outlets (default: a single, generic
        :py:class:`Outlet` with no declared kinds/attributes; concrete
        subclasses normally override this).
        '''
        return [Outlet()]

    def _update_mappings_specific(
            self,
            added: tSet[CodesNSLCE],
            removed: tSet[CodesNSLCE]) -> bool:

        '''
        Update :py:attr:`~BaseOperator._mappings` for added/removed input
        codes, via :py:attr:`codes_projection` (see
        :py:meth:`BaseOperator.update_mappings`).

        :param added:
            Newly available input codes.
        :type added:
            set of :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :param removed:
            No-longer-available input codes.
        :type removed:
            set of :py:class:`~pyrocko.squirrel.model.CodesNSLCE`

        :rtype:
            bool
        '''

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
    '''
    Removes the instrument response, restituting to a chosen physical
    quantity.

    Requires ``'waveform'`` and ``'response'`` from its input and
    provides ``'channel'``/``'waveform'``. Output codes get an
    ``R<quantity-letter>`` suffix appended to the channel's extra code
    (e.g. ``RV`` for velocity).
    '''

    name = String.T(default='rest')
    codes_projection = basic_codes_projection_t(
        '{i.network}.{i.station}.{i.location}.{i.channel}'
        '.{i.extra}R{o.quantity}')
    quantity = QuantityType.T(
        default='velocity',
        help='Physical quantity to restitute to.')
    frequency_min = Float.T(
        help='Lower corner frequency of the deconvolution passband [Hz].')
    frequency_max = Float.T(
        help='Upper corner frequency of the deconvolution passband [Hz].')
    frequency_taper_factor = Float.T(
        default=1.5,
        help='Factor by which ``frequency_min``/``frequency_max`` are '
             'divided/multiplied to get the default taper corner '
             'frequencies, if not given explicitly.')
    frequency_taper_min = Float.T(
        optional=True,
        help='Lower taper corner frequency [Hz], overriding the default '
             'derived from ``frequency_min``/``frequency_taper_factor``.')
    frequency_taper_max = Float.T(
        optional=True,
        help='Upper taper corner frequency [Hz], overriding the default '
             'derived from ``frequency_max``/``frequency_taper_factor``.')
    time_taper_factor = Float.T(
        default=2.0,
        help='Factor controlling the time-domain taper length and, via '
             '``time_padding_extra_factor``, the input time padding '
             'requested from upstream (see :py:meth:`get_time_padding`).')
    time_padding_extra_factor = Float.T(
        default=2.0,
        help='Extra safety factor applied on top of ``time_taper_factor`` '
             'when computing the input time padding requested from '
             'upstream (see :py:meth:`get_time_padding`).')

    @property
    def kind_provides(self):
        return ('channel', 'waveform')

    @property
    def kind_requires(self):
        return ('waveform', 'response')

    def get_outlets(self) -> tList[Outlet]:
        '''
        Get outlets: a single one, tagged with the restituted quantity's
        first letter (used as ``{o.quantity}`` in
        :py:attr:`codes_projection`).
        '''
        return [Outlet(
            kinds=['channel', 'waveform'],
            attributes=dict(quantity=self.quantity[0]))]

    def get_time_padding(self) -> float:
        '''
        Get input time padding needed for the deconvolution's time-domain
        taper, derived from :py:attr:`frequency_min`,
        :py:attr:`time_taper_factor` and
        :py:attr:`time_padding_extra_factor`.
        '''
        return self.time_taper_factor \
            / self.frequency_min * self.time_padding_extra_factor

    def get_taper_frequencies(self):
        '''
        Get the four corner frequencies of the deconvolution's frequency
        taper, as ``(f1, f2, f3, f4)`` (see
        :py:meth:`pyrocko.trace.Trace.transfer`'s ``freqlimits``).

        :rtype:
            :py:class:`tuple` of 4 :py:class:`float`
        '''
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
                codes_to_traces: dict[CodesNSLCE, tList[Trace]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        '''
        Deconvolve the instrument response from each input trace and
        restitute to :py:attr:`quantity`, skipping (with a logged
        warning) traces with missing, ambiguous, or too-low-sample-rate
        responses, or which raise
        :py:exc:`~pyrocko.trace.TraceTooShort`,
        :py:exc:`~pyrocko.trace.NoData` or
        :py:exc:`~pyrocko.response.InvalidResponseError` during
        deconvolution.

        Parameters as for
        :py:meth:`~BaseOperator.process_channels`, with
        ``codes_to_traces`` in place of ``codes_to_channels``.
        '''

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


class Transform(Operator):
    '''
    Base class for coordinate-system transforms (see :py:class:`ToENZ`,
    :py:class:`ToTRZ`, :py:class:`ToLQT`), rotating each group of
    matching channels (via :py:meth:`~pyrocko.squirrel.model.Sensor.\
project_to_enz`-alike sensor projections) into a new component system.

    Requires and provides ``'channel'``/``'waveform'``. Output codes get
    the component character replaced by the target system's component
    (e.g. ``E``/``N``/``Z``) and a ``T<system>`` suffix appended to the
    extra code (e.g. ``TENZ``).
    '''

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
        '''
        Get outlets: one per target :py:attr:`components` character,
        tagged with the target system (lower-cased ``components``, used
        as ``{o.system}``) and that component (used as ``{o.component}``)
        in :py:attr:`codes_projection`.
        '''
        return [
            Outlet(
                kinds=['channel', 'waveform'],
                attributes=dict(
                    system=self.components.lower(),
                    component=c))
            for c in self.components]

    def get_component_codes(self, mapping):
        '''
        Get output codes of a mapping by target component character.

        :param mapping:
            Codes mapping of this operator.
        :type mapping:
            :py:class:`CodesMapping`

        :returns:
            Output codes, keyed by component character.
        :rtype:
            dict
        '''

        return dict(
            (outlet.attributes['component'],
             self.codes_projection._project_single(
                 self, mapping.in_codes[0], outlet))
            for outlet in self.get_outlets())

    def process_channels(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_channels: dict[CodesNSLCE, tList[Channel]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Channel]:

        '''
        Derive rotated output channels, one triplet per input sensor
        (group of matching input channels), with azimuth/dip set from
        :py:meth:`get_orientation`.

        Parameters as for :py:meth:`~BaseOperator.process_channels`.
        '''

        channels_out = []
        for mapping in mappings:
            channels = []
            for in_codes in mapping.in_codes:
                try:
                    channels.extend(codes_to_channels[in_codes])
                except KeyError:
                    pass

            sensors = Sensor.from_channels(channels)
            component_codes = self.get_component_codes(mapping)
            for sensor in sensors:
                for component in self.components:
                    out_codes = component_codes[component]

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
                codes_to_traces: dict[CodesNSLCE, tList[Trace]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        '''
        Rotate each mapping's input traces (which must form exactly one
        sensor's worth of channels) into the target component system via
        :py:meth:`project`, skipping (with a logged warning) mappings
        whose input codes don't group into exactly one sensor.

        Parameters as for :py:meth:`~BaseOperator.process_channels`, with
        ``codes_to_traces`` in place of ``codes_to_channels``.
        '''

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

                codes_mapping = self.get_component_codes(mapping)

                trs_sensor_out = self.project(sensor, trs_sensor)
                for tr in trs_sensor_out:
                    tr.set_codes(*codes_mapping[tr.channel[-1]])

                traces_out.extend(trs_sensor_out)

        return traces_out


class ToENZ(Transform):
    '''
    Rotate sensors to East-North-Up components.

    See :py:class:`Transform` for what's required/provided and how
    output codes are derived.
    '''

    name = String.T(default='enz')
    components = 'ENZ'

    def project(self, sensor, trs_sensor):
        '''
        Rotate a sensor's traces to ENZ, see
        :py:meth:`~pyrocko.squirrel.model.Sensor.project_to_enz`.
        '''
        return sensor.project_to_enz(trs_sensor)

    def get_orientation(self, sensor, component):
        '''
        Get ``(azimuth, dip)`` for a given ENZ component character.
        '''
        return {
            'E': (90., 0.),
            'N': (0., 0.),
            'Z': (0., -90.)}[component]


class ToTRZ(Transform):
    '''
    Rotate sensors to Transverse-Radial-Up components, relative to a
    source or a fixed azimuth.

    See :py:class:`Transform` for what's required/provided and how
    output codes are derived.
    '''

    name = String.T(default='trz')
    components = 'TRZ'
    origin = Location.T(
        optional=True,
        help='Source location to compute the backazimuth from. Required '
             'unless ``azimuth`` is given directly.')
    azimuth = Float.T(
        optional=True,
        help='Azimuth of the sensor as seen from the source [deg], '
             'overriding the automatic backazimuth computation from '
             '``origin``.')

    def project(self, sensor, trs_sensor):
        '''
        Rotate a sensor's traces to TRZ, see
        :py:meth:`~pyrocko.squirrel.model.Sensor.project_to_trz`.
        '''
        return sensor.project_to_trz(
            source=self.origin,
            traces=trs_sensor,
            azimuth=self.azimuth)

    def get_orientation(self, sensor, component):
        '''
        Get ``(azimuth, dip)`` for a given TRZ component character.
        '''
        if self.azimuth is not None:
            azimuth = self.azimuth
        else:
            azimuth = self.origin.azibazi_to(sensor)[1] + 180

        return {
            'T': ((azimuth + 90 + 180.) % 360. - 180., 0.),
            'R': ((azimuth + 180.) % 360. - 180., 0.),
            'Z': (0, -90.)}[component]


class ToLQT(Transform):
    '''
    Rotate sensors to Longitudinal-Transverse-SH (ray-based)
    components, relative to a source and earth model.

    See :py:class:`Transform` for what's required/provided and how
    output codes are derived.
    '''

    name = String.T(default='lqt')
    components = 'LQT'
    origin = Location.T(
        optional=True,
        help='Source location to compute backazimuth/incidence angle '
             'from. Required unless ``azimuth``/``incidence`` (for '
             ':py:meth:`get_orientation`) and ``distance`` (for '
             ':py:meth:`project`) are given directly.')
    earthmodel = Earthmodel1D.T(
        optional=True,
        help='Earth model used for automatic incidence angle '
             'computation. Defaults to a built-in global model, see '
             ':py:func:`~pyrocko.squirrel.model.get_default_earthmodel`.')
    phases = List.T(
        String.T(),
        help='Candidate seismic phases to use for automatic incidence '
             'angle computation, in Cake phase syntax.')
    distance = Float.T(
        optional=True,
        help='Source-sensor distance [deg], overriding the automatic '
             'computation from ``origin``, for use by :py:meth:`project`.')
    source_depth = Float.T(
        optional=True,
        help='Source depth [m], used for automatic incidence angle '
             'computation by :py:meth:`project` (unlike '
             ':py:meth:`get_orientation`, which always uses ``origin``\'s '
             'depth, see class docstring caveat).')
    azimuth = Float.T(
        optional=True,
        help='Azimuth of the sensor as seen from the source [deg], for '
             'use by :py:meth:`get_orientation` (:py:meth:`project` '
             'always computes azimuth automatically, see class '
             'docstring caveat).')
    incidence = Float.T(
        optional=True,
        help='Incidence angle [deg], for use by '
             ':py:meth:`get_orientation` (:py:meth:`project` always '
             'computes incidence automatically, see class docstring '
             'caveat).')

    def project(self, sensor, trs_sensor):
        '''
        Rotate a sensor's traces to LQT, see
        :py:meth:`~pyrocko.squirrel.model.Sensor.project_to_lqt`.

        Unlike :py:meth:`get_orientation`, this always lets
        ``source_depth``/``azimuth``/``incidence`` be computed
        automatically from ``origin``/``earthmodel``/``phases`` rather
        than using this operator's own same-named attributes (see class
        docstring).
        '''

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
        '''
        Get ``(azimuth, dip)`` for a given LQT component character,
        computing azimuth/incidence automatically from ``origin``/
        ``earthmodel``/``phases`` unless overridden via ``azimuth``/
        ``incidence``.
        '''
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
    '''
    Sum (or average) all input carpets into a single combined carpet.

    Requires and provides ``'carpet'``. Its
    :py:attr:`~Operator.codes_projection` has no ``{i.*}``/``{o.*}``
    placeholders, so *all* input carpets are grouped into one and summed
    together, regardless of station/network — e.g. to stack a whole
    array's carpets into one. Output codes are fixed to
    ``'.SUM...'`` (empty network/location/channel/extra, station
    ``'SUM'``).
    '''

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
        '''
        Get outlets: a single one, tagged ``'carpet'``.
        '''
        return [Outlet(kinds=['carpet'])]

    def process_carpets(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_carpets: dict[CodesNSLCE, tList[Carpet]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        '''
        Sum each mapping's input carpets (see
        :py:func:`pyrocko.carpet.sum`), controlled by :py:attr:`nan_aware`
        /:py:attr:`normalize`, skipping (with a logged warning) mappings
        where summation fails (e.g. incompatible carpet geometries).

        Parameters as for :py:meth:`~BaseOperator.process_channels`, with
        ``codes_to_carpets`` in place of ``codes_to_channels``.
        '''

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
    '''
    Combine horizontal (N/E) and vertical (Z) component carpets into a
    single horizontal-to-vertical ratio carpet, per sensor.

    Requires and provides ``'carpet'``. The input carpets must contain
    logarithmic values, as produced by the spectrogram operators (see
    :py:mod:`pyrocko.squirrel.operators.spectrogram`), whose carpets hold
    the natural logarithm of the power spectrum. The horizontal components
    are averaged, as are the vertical ones, and the vertical average is
    subtracted from the horizontal one. In the logarithmic domain, the
    average corresponds to a geometric mean and the subtraction to a
    ratio, so for spectrogram input, the result is

    .. math::

        \\ln \\frac{\\sqrt{P_N P_E}}{P_Z},

    the natural logarithm of the H/V power ratio, or twice the natural
    logarithm of the H/V amplitude ratio. Output codes get an ``HV``
    suffix appended to the channel's extra code, with the component
    character dropped.
    '''

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
        '''
        Get outlets: a single one, tagged ``'carpet'``.
        '''
        return [Outlet(kinds=['carpet'])]

    def process_carpets(
                self,
                mappings: tList[CodesMapping],
                in_codes: tList[CodesNSLCE],
                codes_to_carpets: dict[CodesNSLCE, tList[Carpet]],
                tmin: TimeFloat = None,
                tmax: TimeFloat = None,
            ) -> tList[Trace]:

        '''
        Compute the H/V combination for each mapping's input carpets,
        splitting them by component character (``'N'``/``'E'`` vs.
        ``'Z'``), averaging each group (see :py:func:`pyrocko.carpet.\
sum`) and subtracting the vertical average from the horizontal one. The
        input values must be logarithmic, see :py:class:`CarpetHOverV`.
        Mappings where this fails (e.g. incompatible carpet geometries, or
        a missing component) are skipped with a logged warning.

        Parameters as for :py:meth:`~BaseOperator.process_channels`, with
        ``codes_to_carpets`` in place of ``codes_to_channels``.
        '''

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
    'ToENZ',
    'ToTRZ',
    'ToLQT',
    'CarpetSum',
    'CarpetHOverV',
]
