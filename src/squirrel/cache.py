# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import logging

logger = logging.getLogger('pyrocko.squirrel.cache')


class ContentCache(object):

    def __init__(self):
        self._entries = {}
        self._accessor_ticks = {}

    def advance_accessor(self, accessor):
        if accessor not in self._accessor_ticks:
            self._accessor_ticks[accessor] = 0

        ta = self._accessor_ticks[accessor]

        delete = []
        for path_segment, entry in self._entries.items():
            t = entry[2].get(accessor, ta)
            if t < ta:
                del entry[2][accessor]
                if not entry[2]:
                    delete.append(path_segment)

        for path_segment in delete:
            logger.debug('Forgetting (advance): %s %s' % path_segment)
            del self._entries[path_segment]

        self._accessor_ticks[accessor] += 1

    def clear_accessor(self, accessor):
        delete = []
        for path_segment, entry in self._entries.items():
            entry[2].pop(accessor, None)
            if not entry[2]:
                delete.append(path_segment)

        for path_segment in delete:
            logger.debug('Forgetting (clear): %s %s' % path_segment)
            del self._entries[path_segment]

        del self._accessor_ticks[accessor]

    def clear(self):
        for accessor in list(self._accessor_ticks.keys()):
            self.clear_accessor(accessor)

        self._entries = {}
        self._accessor_ticks = {}

    def has(self, nut):
        path, segment, element, nut_mtime = nut.key

        try:
            cache_mtime = self._entries[path, segment][0]
        except KeyError:
            return False

        return cache_mtime == nut_mtime

    def get(self, nut, accessor='default'):
        path, segment, element, mtime = nut.key
        entry = self._entries[path, segment]

        if accessor not in self._accessor_ticks:
            self._accessor_ticks[accessor] = 0

        entry[2][accessor] = self._accessor_ticks[accessor]

        return entry[1][element]

    def _prune_outdated(self, path, segment, nut_mtime):
        try:
            cache_mtime = self._entries[path, segment][0]
        except KeyError:
            return

        if cache_mtime != nut_mtime:
            logger.debug('Forgetting (outdated): %s %s' % (path, segment))
            del self._entries[path, segment]

    def put(self, nut):
        path, segment, element, mtime = nut.key
        self._prune_outdated(path, segment, nut.file_mtime)

        if (path, segment) not in self._entries:
            self._entries[path, segment] = nut.file_mtime, {}, {}

        self._entries[path, segment][1][element] = nut.content
