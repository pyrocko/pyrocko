# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import os.path as op
import logging
import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

from pyrocko import util
from pyrocko.guts import String, Dict, Duration, dump_all

from .base import Source
from ..model import ehash
from ..lock import LockDir

logger = logging.getLogger('pyrocko.squirrel.client.catalog')


class Link(object):
    def __init__(self, tmin, tmax, tmodified, nevents=-1, content_id=None):
        self.tmin = tmin
        self.tmax = tmax
        self.tmodified = tmodified
        self.nevents = nevents
        self.content_id = content_id

    def __str__(self):
        return 'span %s - %s, access %s, nevents %i' % (
            util.tts(self.tmin),
            util.tts(self.tmax),
            util.tts(self.tmodified),
            self.nevents)


class NoSuchCatalog(Exception):
    pass


def get_catalog(name):
    if name == 'geofon':
        from pyrocko.client.geofon import Geofon
        return Geofon()
    elif name == 'gcmt':
        from pyrocko.client.globalcmt import GlobalCMT
        return GlobalCMT()
    else:
        raise NoSuchCatalog(name)


class CatalogSource(Source):

    catalog = String.T()
    query_args = Dict.T(String.T(), String.T(), optional=True)
    expires = Duration.T(optional=True)
    anxious = Duration.T(optional=True)
    cache_path = String.T(optional=True)

    def __init__(self, catalog, query_args=None, **kwargs):
        Source.__init__(self, catalog=catalog, query_args=query_args, **kwargs)

        self._hash = self.make_hash()
        self._nevents_query_hint = 1000
        self._nevents_chunk_hint = 5000
        self._tquery = 3600.*24.
        self._tquery_limits = (3600., 3600.*24.*365.)

    def setup(self, squirrel):
        self._force_query_age_max = self.anxious
        self._catalog = get_catalog(self.catalog)

        self._cache_path = op.join(
            self.cache_path or squirrel._cache_path,
            'catalog',
            self.get_hash())

        util.ensuredir(self._cache_path)

    def make_hash(self):
        s = self.catalog
        if self.query_args is not None:
            s += ','.join(
                '%s:%s' % (k, self.query_args[k])
                for k in sorted(self.query_args.keys()))
        else:
            s += 'noqueryargs'

        return ehash(s)

    def get_hash(self):
        return self._hash

    def update_event_inventory(self, squirrel, constraint=None):

        with LockDir(self._cache_path):
            self._load_chain()

            assert constraint is not None
            if constraint is not None:
                tmin, tmax = constraint.tmin, constraint.tmax

            tmin_sq, tmax_sq = squirrel.get_time_span()

            if tmin is None:
                tmin = tmin_sq

            if tmax is None:
                tmax = tmax_sq

            if tmin is None or tmax is None:
                logger.warn(
                    'Cannot query catalog source "%s" without time '
                    'constraint. Could not determine appropriate time '
                    'constraint from current data holdings (no data?).'
                    % self.catalog)

                return

            if tmin >= tmax:
                return

            tnow = time.time()
            modified = False

            if not self._chain:
                self._chain = [Link(tmin, tmax, tnow)]
                modified = True
            else:
                if tmin < self._chain[0].tmin:
                    self._chain[0:0] = [Link(tmin, self._chain[0].tmin, tnow)]
                    modified = True
                if self._chain[-1].tmax < tmax:
                    self._chain.append(Link(self._chain[-1].tmax, tmax, tnow))
                    modified = True

            chain = []
            remove = []
            for link in self._chain:
                if tmin < link.tmax and link.tmin < tmax \
                        and self._outdated(link, tnow):

                    if link.content_id:
                        remove.append(
                            self._get_events_file_path(link.content_id))

                    tmin_query = max(link.tmin, tmin)
                    tmax_query = min(link.tmax, tmax)

                    if link.tmin < tmin_query:
                        chain.append(Link(link.tmin, tmin_query, tnow))

                    if tmin_query < tmax_query:
                        for link in self._iquery(tmin_query, tmax_query, tnow):
                            chain.append(link)

                    if tmax_query < link.tmax:
                        chain.append(Link(tmax_query, link.tmax, tnow))

                    modified = True

                else:
                    chain.append(link)

            if modified:
                self._chain = chain
                self._dump_chain()
                squirrel.remove(remove)

            add = []
            for link in self._chain:
                if link.content_id:
                    add.append(self._get_events_file_path(link.content_id))

            squirrel.add(add, kinds=['event'], format='yaml')

    def _iquery(self, tmin, tmax, tmodified):

        nwant = self._nevents_query_hint
        tlim = self._tquery_limits

        t = tmin
        tpack_min = tmin

        events = []
        while t < tmax:
            tmin_query = t
            tmax_query = min(t + self._tquery, tmax)

            events_new = self._query(tmin_query, tmax_query)
            nevents_new = len(events_new)
            events.extend(events_new)
            while len(events) > int(self._nevents_chunk_hint * 1.5):
                tpack_max = events[self._nevents_chunk_hint].time
                yield self._pack(
                    events[:self._nevents_chunk_hint],
                    tpack_min, tpack_max, tmodified)

                tpack_min = tpack_max
                events[:self._nevents_query_hint] = []

            t += self._tquery

            if tmax_query != tmax:
                if nevents_new < 5:
                    self._tquery *= 10.0

                elif not (nwant // 2 < nevents_new < nwant * 2):
                    self._tquery /= float(nevents_new) / float(nwant)

                self._tquery = max(tlim[0], min(self._tquery, tlim[1]))

        if self._force_query_age_max is not None:
            tsplit = tmodified - self._force_query_age_max
            if tpack_min < tsplit < tmax:
                events_older = []
                events_newer = []
                for ev in events:
                    if ev.time < tsplit:
                        events_older.append(ev)
                    else:
                        events_newer.append(ev)

                yield self._pack(events_older, tpack_min, tsplit, tmodified)
                yield self._pack(events_newer, tsplit, tmax, tmodified)
                return

        yield self._pack(events, tpack_min, tmax, tmodified)

    def _pack(self, events, tmin, tmax, tmodified):
        if events:
            content_id = ehash(
                self.get_hash() + ' %r %r %r' % (tmin, tmax, tmodified))
            path = self._get_events_file_path(content_id)
            dump_all(events, filename=path)
        else:
            content_id = None

        return Link(tmin, tmax, tmodified, len(events), content_id)

    def _query(self, tmin, tmax):
        logger.info('Querying catalog "%s" for time span %s - %s.' % (
            self.catalog, util.tts(tmin), util.tts(tmax)))

        return self._catalog.get_events(
            (tmin, tmax),
            **(self.query_args or {}))

    def _outdated(self, link, tnow):
        if link.nevents == -1:
            return True

        if self._force_query_age_max \
                and link.tmax + self._force_query_age_max > link.tmodified:

            return True

        if self.expires is not None \
                and link.tmodified < tnow - self.expires:

            return True

        return False

    def _get_events_file_path(self, fhash):
        return op.join(self._cache_path, fhash + '.pf')

    def _get_chain_file_path(self):
        return op.join(self._cache_path, 'chain.pickle')

    def _load_chain(self):
        path = self._get_chain_file_path()
        if op.exists(path):
            with open(path, 'rb') as f:
                self._chain = pickle.load(f)
        else:
            self._chain = []

    def _dump_chain(self):
        with open(self._get_chain_file_path(), 'wb') as f:
            pickle.dump(self._chain, f, protocol=2)


__all__ = [
    'CatalogSource'
]
