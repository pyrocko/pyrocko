# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

import math

import numpy as num

from pyrocko.guts import Object, Float, StringChoice, Int, String, Bool, \
    Tuple, List

from pyrocko.gui import talkie


class Color(Object):
    r = Float.T(default=0.0)
    g = Float.T(default=0.0)
    b = Float.T(default=0.0)
    a = Float.T(default=1.0)

    @property
    def qt_color(self):
        from pyrocko.gui.qt_compat import qg
        color = qg.QColor(*(int(round(x*255)) for x in (
            self.r, self.g, self.b, self.a)))
        return color


class ScalingMode(StringChoice):
    choices = ['same', 'individual', 'fixed']


class ScalingBase(StringChoice):
    choices = [
        'min-max', 'mean-plusminus-1-sigma',
        'mean-plusminus-2-sigma', 'mean-plusminus-4-sigma']


class Filter(Object):

    def apply(self, tr):
        pass

    def tpad(self):
        return 0.0


class Demean(Filter):
    def apply(self, tr):
        tr.ydata = tr.ydata - num.mean(tr.ydata)


class ButterLowpass(Filter):
    order = Int.T(default=4)
    corner = Float.T()
    pad_factor = Float.T(default=1.0, optional=True)

    def apply(self, tr):
        tr.lowpass(self.order, self.corner)

    def tpad(self):
        return self.pad_factor/self.corner


class ButterHighpass(Filter):
    order = Int.T(default=4)
    corner = Float.T()
    pad_factor = Float.T(default=1.0, optional=True)

    def apply(self, tr):
        tr.highpass(self.order, self.corner)

    def tpad(self):
        return self.pad_factor/self.corner


class Downsample(Filter):
    deltat = Float.T()

    def apply(self, tr):
        tr.downsample_to(self.deltat)


class TextStyle(talkie.Talkie):
    family = String.T(default='default', optional=True)
    size = Float.T(default=9.0, optional=True)
    bold = Bool.T(default=False, optional=True)
    italic = Bool.T(default=False, optional=True)
    color = Color.T(default=Color.D())
    outline = Bool.T(default=False)
    background_color = Color.T(optional=True)

    @property
    def qt_font(self):
        from pyrocko.gui.qt_compat import qg
        font = qg.QFont(self.family)
        font.setPointSizeF(self.size)
        font.setBold(self.bold)
        font.setItalic(self.italic)
        return font


class Style(talkie.Talkie):
    antialiasing = Bool.T(default=False, optional=True)
    label_textstyle = TextStyle.T(default=TextStyle.D(
        bold=True,
        background_color=Color(r=1.0, g=1.0, b=1.0, a=0.5),
        ))
    title_textstyle = TextStyle.T(default=TextStyle.D(bold=True, size=12.0))
    marker_textstyle = TextStyle.T(default=TextStyle.D(
        bold=False,
        size=9.0,
        italic=True,
        background_color=Color(r=1.0, g=1.0, b=0.7),
        outline=True,
        ))
    marker_color = Color.T(default=Color.D())
    trace_resolution = Float.T(default=2.0, optional=True)
    trace_color = Color.T(default=Color.D())
    background_color = Color.T(default=Color.D(r=1.0, g=1.0, b=1.0))


class Scaling(talkie.Talkie):
    mode = ScalingMode.T(default='same')
    base = ScalingBase.T(default='min-max')
    min = Float.T(default=-1.0, optional=True)
    max = Float.T(default=1.0, optional=True)
    gain = Float.T(default=1.0, optional=True)


class State(talkie.TalkieRoot):
    nslc = Tuple.T(4, String.T(default=''))
    tline = Float.T(default=60.*60.)
    nlines = Int.T(default=24)
    iline = Int.T(default=0)

    follow = Bool.T(default=False)

    style = Style.T(default=Style.D())
    filters = List.T(Filter.T())
    scaling = Scaling.T(default=Scaling.D())

    npages_cache = Int.T(default=10, optional=True)

    @property
    def tmin(self):
        return self.iline*self.tline

    @tmin.setter
    def tmin(self, tmin):
        self.iline = int(math.floor(tmin / self.tline))

    @property
    def tmax(self):
        return (self.iline+self.nlines)*self.tline

    @tmax.setter
    def tmax(self, tmax):
        self.iline = int(math.ceil(tmax / self.tline))-self.nlines
