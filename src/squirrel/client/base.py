# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from pyrocko.guts import Object, Timestamp


guts_prefix = 'squirrel'


class Constraint(Object):

    '''
    Used by some data-sources to grow or join locally mirrored data selections.

    Squirrel data-sources typically try to mirror only a subset of the remotely
    available data. This subset may need to be grown or updated when data from
    other time intervals or from locations outside the initially requested
    region is requested. This class helps in the involved bookeeping.

    The current implementation only supports a time interval selection with a
    single time span but more sophisticated constraints, including e.g.
    location boxes could be thought of.
    '''

    tmin = Timestamp.T(optional=True)
    tmax = Timestamp.T(optional=True)

    def contains(self, constraint):
        '''
        Check if the constraint completely includes a more restrictive one.

        :param constraint:
            Other constraint.
        :type constraint:
            :py:class:`Constraint`
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

        :param constraint:
            Other constraint.
        :type constraint:
            :py:class:`Constraint`

        Update is done in-place.
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

    '''
    Base class for Squirrel data-sources.

    Data-sources can be attached to a Squirrel instance to allow transparent
    access to remote (or otherwise generated) resources, e.g. through FDSN web
    services (:py:class:`~pyrocko.squirrel.client.fdsn.FDSNSource`) or online
    event catalogs (:py:class:`~pyrocko.client.catalog.CatalogSource`).

    Derived classes implement the details of querying, caching, updating and
    bookkeeping of the accessed data.
    '''

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
