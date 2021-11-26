# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import logging
import re
import fnmatch

from ..model import QuantityType, separator
from .. import error

from pyrocko.guts import Object, String, Duration, Float, clone

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
            d = r'[^%s]*' % separator
        elif c == '?':
            d = r'[^%s]' % separator
        elif c == '.':
            d = separator
        else:
            d = c

        dd.append(d)
    reg = ''.join(dd)
    return reg


_compiled_patterns = {}


def compiled(pattern):
    if pattern not in _compiled_patterns:
        rpattern = re.compile(fnmatch.translate(pattern), re.I)
        _compiled_patterns[pattern] = rpattern
    else:
        rpattern = _compiled_patterns[pattern]

    return rpattern


class Filtering(Object):

    def filter(self, it):
        return list(it)


class RegexFiltering(Object):
    pattern = String.T(default=r'(.*)')

    def __init__(self, **kwargs):
        Filtering.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def filter(self, it):
        return [
            x for x in it if self._compiled_pattern.fullmatch(x)]


class Grouping(Object):

    def key(self, codes):
        return codes


class RegexGrouping(Grouping):
    pattern = String.T(default=r'(.*)')

    def __init__(self, **kwargs):
        Grouping.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def key(self, codes):
        return self._compiled_pattern.fullmatch(codes).groups()


class ComponentGrouping(RegexGrouping):
    pattern = String.T(default=_cglob_translate('(*.*.*.*.*)?(.*)'))


class Naming(Object):
    suffix = String.T(default='')

    def get_name(self, base):
        return base + self.suffix


class RegexNaming(Naming):
    pattern = String.T(default=r'(.*)')
    replacement = String.T(default=r'\1')

    def __init__(self, **kwargs):
        Naming.__init__(self, **kwargs)
        self._compiled_pattern = re.compile(self.pattern)

    def get_name(self, base):
        return self._compiled_pattern.sub(
            self.replacement, base) + self.suffix


class ReplaceComponentNaming(RegexNaming):
    pattern = String.T(default=_cglob_translate('(*.*.*.*.*)?(.*)'))
    replacement = String.T(default=r'\1{component}\2')


def lsplit_codes(lcodes):
    return [tuple(codes.split(separator)) for codes in lcodes]


class Operator(Object):

    filtering = Filtering.T(default=Filtering.D())
    grouping = Grouping.T(default=Grouping.D())
    naming = Naming.T(default=Naming.D())

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._output_names_cache = {}
        self._groups = {}
        self._available = []

    @property
    def name(self):
        return self.__class__.__name__

    def describe(self):
        return self.name

    def iter_mappings(self):
        for k, group in self._groups.items():
            if group[1] is None:
                group[1] = sorted(group[0])

            yield (
                lsplit_codes(group[1]),
                lsplit_codes(group[2]))

    def update_mappings(self, available, registry):
        available = list(available)
        removed, added = odiff(self._available, available)

        filt = self.filtering.filter
        gkey = self.grouping.key
        groups = self._groups

        need_update = set()

        def deregister(group):
            for codes in group[2]:
                del registry[codes]

        def register(group):
            for codes in group[2]:
                if codes in registry:
                    logger.warn('duplicate operator output codes: %s' % codes)
                registry[codes] = (self, group)

        for codes in filt(removed):
            k = gkey(codes)
            groups[k][0].remove(codes)
            need_update.add(k)

        for codes in filt(added):
            k = gkey(codes)
            if k not in groups:
                groups[k] = [set(), None, ()]

            groups[k][0].add(codes)
            need_update.add(k)

        for k in need_update:
            group = groups[k]
            deregister(group)
            group[1] = tuple(sorted(group[0]))
            if not group[1]:
                del groups[k]
            else:
                group[2] = self._out_codes(group[1])
                register(group)

        self._available = available

    def _out_codes(self, in_codes):
        return [self.naming.get_name(codes) for codes in in_codes]

    def get_channels(self, squirrel, group, *args, **kwargs):
        _, in_codes, out_codes = group
        assert len(in_codes) == 1 and len(out_codes) == 1
        in_codes_tup = in_codes[0].split(separator)
        channels = squirrel.get_channels(codes=in_codes_tup, **kwargs)
        agn, net, sta, loc, cha, ext = out_codes[0].split(separator)
        channels_out = []
        for channel in channels:
            channel_out = clone(channel)
            channel_out.set_codes(
                agency=agn,
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                extra=ext)
            channels_out.append(channel_out)

        return channels_out

    def get_waveforms(self, squirrel, group, **kwargs):
        _, in_codes, out_codes = group
        assert len(in_codes) == 1 and len(out_codes) == 1

        in_codes_tup = in_codes[0].split(separator)
        trs = squirrel.get_waveforms(codes=in_codes_tup, **kwargs)
        agn, net, sta, loc, cha, ext = out_codes[0].split(separator)
        for tr in trs:
            tr.set_codes(
                agency=agn,
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                extra=ext)

        return trs

    # def update_waveforms(self, squirrel, tmin, tmax, codes):
    #     if codes is None:
    #         for _, in_codes, out_codes in self._groups.values():
    #             for codes in


class Parameters(Object):
    pass


class RestitutionParameters(Parameters):
    frequency_min = Float.T()
    frequency_max = Float.T()
    frequency_taper_factor = Float.T(default=1.5)
    time_taper_factor = Float.T(default=2.0)


class Restitution(Operator):
    naming = Naming(suffix='R{quantity}')
    quantity = QuantityType.T(default='displacement')

    @property
    def name(self):
        return 'Restitution(%s)' % self.quantity[0]

    def _out_codes(self, group):
        return [
            self.naming.get_name(codes).format(
                quantity=self.quantity[0])
            for codes in group]

    def get_tpad(self, params):
        return params.time_taper_factor / params.frequency_min

    def get_waveforms(
            self, squirrel, codes, params, tmin, tmax, **kwargs):

        self_, in_codes, out_codes = squirrel.get_operator_group(codes)
        assert self is self_
        assert len(in_codes) == 1 and len(out_codes) == 1
        in_codes_tup = tuple(in_codes[0].split(separator))

        tpad = self.get_tpad(params)

        tmin_raw = tmin - tpad
        tmax_raw = tmax + tpad

        trs = squirrel.get_waveforms(
            codes=in_codes_tup, tmin=tmin_raw, tmax=tmax_raw, **kwargs)

        try:
            resp = squirrel.get_response(
                codes=in_codes_tup,
                tmin=tmin_raw,
                tmax=tmax_raw).get_effective(self.quantity)

        except error.NotAvailable:
            return []

        freqlimits = (
            params.frequency_min / params.frequency_taper_factor,
            params.frequency_min,
            params.frequency_max,
            params.frequency_max * params.frequency_taper_factor)

        agn, net, sta, loc, cha, ext = out_codes[0].split(separator)
        trs_rest = []
        for tr in trs:
            tr_rest = tr.transfer(
                tfade=tpad,
                freqlimits=freqlimits,
                transfer_function=resp,
                invert=True)

            tr_rest.set_codes(
                agency=agn,
                network=net,
                station=sta,
                location=loc,
                channel=cha,
                extra=ext)

            trs_rest.append(tr_rest)

        return trs_rest


class Shift(Operator):
    naming = Naming(suffix='S')
    delay = Duration.T()


class Transform(Operator):
    grouping = Grouping.T(default=ComponentGrouping.D())
    naming = ReplaceComponentNaming(suffix='T{system}')

    def _out_codes(self, group):
        return [
            self.naming.get_name(group[0]).format(
                component=c, system=self.components.lower())
            for c in self.components]


class ToENZ(Transform):
    components = 'ENZ'

    def get_waveforms(
            self, squirrel, in_codes, out_codes, params, tmin, tmax, **kwargs):

        trs = squirrel.get_waveforms(
            codes=lsplit_codes(in_codes), tmin=tmin, tmax=tmax, **kwargs)

        for tr in trs:
            print(tr)


class ToRTZ(Transform):
    components = 'RTZ'
    backazimuth = Float.T()


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
    'Operator',
    'RestitutionParameters',
    'Restitution',
    'Shift',
    'ToENZ',
    'ToRTZ',
    'ToLTQ',
    'Composition']
