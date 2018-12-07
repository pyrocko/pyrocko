# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division


class EarthquakeCatalog(object):

    def get_event(self, name):
        raise Exception('This method should be implemented in derived class.')

    def iter_event_names(self, time_range, **kwargs):
        raise Exception('This method should be implemented in derived class.')

    def get_event_names(self, time_range, **kwargs):
        return list(self.iter_event_names(time_range, **kwargs))

    def get_events(self, time_range, **kwargs):
        return list(self.iter_events(time_range, **kwargs))

    def iter_events(self, time_range, **kwargs):
        for name in self.iter_event_names(time_range, **kwargs):
            yield self.get_event(name)


class NotFound(Exception):
    def __init__(self, url=None):
        Exception.__init__(self)
        self._url = url

    def __str__(self):
        if self._url:
            return 'No results for request %s' % self._url
        else:
            return 'No results for request'
