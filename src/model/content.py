# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import math

from pyrocko import util


g_tmin, g_tmax = util.get_working_system_time_range()[:2]


def time_or_none_to_str(x, format):
    if x is None:
        return '...'
    else:
        return util.time_to_str(x, format=format)


def squirrel_content(cls):

    def str_codes(self):
        return '.'.join(self.codes)

    cls.str_codes = property(str_codes)

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

    cls.str_time_span = property(str_time_span)

    def summary(self):
        return '%s %-16s %s' % (
            self.__class__.__name__, self.str_codes, self.str_time_span)

    if not hasattr(cls, 'summary'):
        cls.summary = property(summary)

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    cls.__lt__ = __lt__

    def __key__(self):
        return self.codes, self.time_span_g_clipped

    cls.__key__ = __key__

    @property
    def time_span_g_clipped(self):
        tmin, tmax = self.time_span
        return (
            tmin if tmin is not None else g_tmin,
            tmax if tmax is not None else g_tmax)

    cls.time_span_g_clipped = property(time_span_g_clipped)

    return cls
