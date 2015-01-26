# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from pyrocko.guts import Object, Timestamp


class Constraint(Object):

    tmin = Timestamp.T(optional=True)
    tmax = Timestamp.T(optional=True)

    def contains(self, constraint):
        '''
        Check if the constraint completely includes a more restrictive one.
        '''

        if self.tmin is not None and constraint.tmin is not None:
            b1 = self.tmin <= constraint.tmin
        elif self.tmin is None:
            b1 = True
        else:
            b1 = False

        if self.tmax is not None and constraint.tmax is not None:
            b2 = constraint.tmax <= self.tmax
        elif self.tmax is None:
            b2 = True
        else:
            b2 = False

        return b1 and b2

    def expand(self, constraint):
        '''
        Widen constraint to include another given constraint.
        '''

        if constraint.tmin is None or self.tmin is None:
            self.tmin = None
        else:
            self.tmin = min(constraint.tmin, self.tmin)

        if constraint.tmax is None or self.tmax is None:
            self.tmax = None
        else:
            self.tmax = max(constraint.tmax, self.tmax)


class Source(Object):

    def update_channel_inventory(self, squirrel, constraint):
        '''
        Let local inventory be up-to-date with remote for a given constraint.
        '''

        pass

    def update_event_inventory(self, squirrel, constraint):
        '''
        Let local inventory be up-to-date with remote for a given constraint.
        '''

        pass

    def update_waveform_promises(self, squirrel, constraint):
        '''
        Let local inventory be up-to-date with remote for a given constraint.
        '''

        pass


__all__ = [
    'Source',
    'Constraint',
]
