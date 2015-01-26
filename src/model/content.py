# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import math

from pyrocko.guts import Object
from pyrocko import util


g_tmin, g_tmax = util.get_working_system_time_range()[:2]


def time_or_none_to_str(x, format):
    if x is None:
        return '...'.ljust(17)
    else:
        return util.time_to_str(x, format=format)


class Content(Object):
    '''
    Base class for Pyrocko content objects.
    '''

    @property
    def str_codes(self):
        return '.'.join(self.codes)

    @property
    def str_time_span(self):
        tmin, tmax = self.time_span
        deltat = getattr(self, 'deltat', 0)
        if deltat > 0:
            fmt = min(9, max(0, -int(math.floor(math.log10(self.deltat)))))
        else:
            fmt = 6

        if tmin == tmax:
            return '%s' % time_or_none_to_str(tmin, fmt)
        else:
            return '%s - %s' % (
                time_or_none_to_str(tmin, fmt), time_or_none_to_str(tmax, fmt))

    @property
    def summary(self):
        return '%s %-16s %s' % (
            self.__class__.__name__, self.str_codes, self.str_time_span)

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __key__(self):
        return self.codes, self.time_span_g_clipped

    @property
    def time_span_g_clipped(self):
        tmin, tmax = self.time_span
        return (
            tmin if tmin is not None else g_tmin,
            tmax if tmax is not None else g_tmax)
