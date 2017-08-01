# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
import sys


class ProgressBar(object):
    def __init__(self, widgets=['progress'], maxval=1, *args, **kwargs):
        self._widgets = widgets
        self._maxval = maxval
        self._val = 0

    def label(self):
        for widget in self._widgets:
            if isinstance(widget, str):
                return widget

    def start(self):
        return self

    def update(self, val):
        self._val = val
        sys.stderr.write('%s  %i/%i %3.0f%%\n' % (
            self.label(),
            self._val,
            self._maxval,
            100.*float(self._val) / float(self._maxval)))

    def finish(self):
        sys.stderr.write('%s done\n' % self.label())


class Bar(object):
    def __init__(self, *args, **kwargs):
        pass


class Percentage(object):
    def __init__(self, *args, **kwargs):
        pass
