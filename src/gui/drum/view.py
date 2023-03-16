# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

import math
import time
import weakref
import numpy as num

from pyrocko.gui.qt_compat import qc, qg, qw

from pyrocko.gui.talkie import Listener
from pyrocko.gui.drum.state import State, TextStyle
from pyrocko.gui_util import make_QPolygonF, PhaseMarker
from pyrocko import trace, util, pile


def lim(a, x, b):
    return min(max(a, x), b)


class Label(object):
    def __init__(
            self, x, y, label_str,
            anchor='BL',
            style=None,
            keep_inside=None,
            head=None):

        if style is None:
            style = TextStyle()

        text = qg.QTextDocument()
        font = style.qt_font
        if font:
            text.setDefaultFont(font)

        color = style.color.qt_color

        text.setDefaultStyleSheet('span { color: %s; }' % color.name())
        text.setHtml('<span>%s</span>' % label_str)

        self.position = x, y
        self.anchor = anchor
        self.text = text
        self.style = style
        self.keep_inside = keep_inside
        if head:
            self.head = head
        else:
            self.head = (0., 0.)

    def draw(self, p):
        s = self.text.size()
        rect = qc.QRectF(0., 0., s.width(), s.height())

        tx, ty = x, y = self.position
        anchor = self.anchor

        pxs = self.text.defaultFont().pointSize()

        oy = self.head[0] * pxs
        ox = self.head[1]/2. * pxs

        if 'L' in anchor:
            tx -= min(ox*2, rect.width()/2.)

        elif 'R' in anchor:
            tx -= rect.width() - min(ox*2., rect.width()/2.)

        elif 'C' in anchor:
            tx -= rect.width()/2.

        if 'B' in anchor:
            ty -= rect.height() + oy

        elif 'T' in anchor:
            ty += oy

        elif 'M' in anchor:
            ty -= rect.height()/2.

        rect.translate(tx, ty)

        if self.keep_inside:
            keep_inside = self.keep_inside
            if rect.top() < keep_inside.top():
                rect.moveTop(keep_inside.top())

            if rect.bottom() > keep_inside.bottom():
                rect.moveBottom(keep_inside.bottom())

            if rect.left() < keep_inside.left():
                rect.moveLeft(keep_inside.left())

            if rect.right() > keep_inside.right():
                rect.moveRight(keep_inside.right())

        poly = None
        if self.head[0] != 0.:
            l, r, t, b = rect.left(), rect.right(), rect.top(), rect.bottom()

            if 'T' in anchor:
                a, b = t, b
            elif 'B' in anchor:
                a, b = b, t
            elif 'M' in anchor:
                assert False, 'label cannot have head with M alignment'

            c1,  c2 = lim(l, x-ox, r),  lim(l, x+ox, r)

            px = (l, c1, x, c2, r, r, l)
            py = (a, a, y, a, a, b, b)

            poly = make_QPolygonF(px, py)

        tx = rect.left()
        ty = rect.top()

        if self.style.outline or self.style.background_color:
            oldpen = p.pen()
            oldbrush = p.brush()
            if not self.style.outline:
                p.setPen(qg.QPen(qc.Qt.NoPen))

            p.setBrush(self.style.background_color.qt_color)
            if poly:
                p.drawPolygon(poly)
            else:
                p.drawRect(rect)

            if self.style.background_color:
                p.fillRect(rect, self.style.background_color.qt_color)

            p.setPen(oldpen)
            p.setBrush(oldbrush)

        p.translate(tx, ty)
        self.text.drawContents(p)
        p.translate(-tx, -ty)


class ProjectXU(object):
    def __init__(self, xr=(0., 1.), ur=(0., 1.)):
        (self.xmin, self.xmax) = xr
        (self.umin, self.umax) = ur
        self.cxu = (self.umax - self.umin) / (self.xmax - self.xmin)

    def u(self, x):
        return self.umin + (x-self.xmin) * self.cxu

    def x(self, u):
        return self.xmin + (u-self.umin) / self.cxu


class ProjectYV(object):
    def __init__(self, yr=(0., 1.), vr=(0., 1.)):
        (self.ymin, self.ymax) = yr
        (self.vmin, self.vmax) = vr
        self.cyv = (self.vmax - self.vmin) / (self.ymax - self.ymin)

    def v(self, y):
        return self.vmin + (y-self.ymin) * self.cyv

    def y(self, v):
        return self.ymin + (v-self.vmin) / self.cyv


class ProjectXYUV(object):
    def __init__(self, xyr=((0., 1.), (0., 1.)), uvr=((0., 1.), (0., 1.))):
        (self.xmin, self.xmax), (self.ymin, self.ymax) = xyr
        (self.umin, self.umax), (self.vmin, self.vmax) = uvr
        self.cxu = (self.umax - self.umin) / (self.xmax - self.xmin)
        self.cyv = (self.vmax - self.vmin) / (self.ymax - self.ymin)

    def u(self, x):
        return self.umin + (x-self.xmin) * self.cxu

    def v(self, y):
        return self.vmin + (y-self.ymin) * self.cyv

    def x(self, u):
        return self.xmin + (u-self.umin) / self.cxu

    def y(self, v):
        return self.ymin + (v-self.vmin) / self.cyv


class PlotEnv(qg.QPainter):
    def __init__(self, *args):
        qg.QPainter.__init__(self, *args)
        self.umin = 0.
        self.umax = 1.
        self.vmin = 0.
        self.vmax = 1.

    def draw_vline(self, project, x, y0, y1):
        u = project.u(x)
        line = qc.QLineF(u, project.v(y0), u, project.v(y1))
        self.drawLine(line)

    def projector(self, xyr):
        return ProjectXYUV(xyr, self.uvrange)

    @property
    def uvrange(self):
        return (self.umin, self.umax), (self.vmin, self.vmax)


def time_fmt_drumline(t):
    ti = int(t)
    if ti % 60 == 0:
        fmt = '%H:%M'
    else:
        fmt = '%H:%M:%S'

    if ti % (3600*24) == 0:
        fmt = fmt + ' %Y-%m-%d'

    return fmt


class Empty(Exception):
    pass


class DrumLine(qc.QObject, Listener):
    def __init__(self, iline, tmin, tmax, traces, state):
        qc.QObject.__init__(self)
        self.traces = traces
        self.tmin = tmin
        self.tmax = tmax
        self.ymin = 0.
        self.ymax = 1.
        self.ymin_data = 0.
        self.ymax_data = 1.
        self.iline = iline
        self._last_mode = None
        self._ydata_cache = {}
        self._time_per_pixel = None
        self.access_counter = 0

        state.add_listener(
            self.listener_no_args(self._empty_cache),
            path='style.trace_resolution')

    def data_range(self, mode='min-max'):
        if not self.traces:
            raise Empty()

        modemap = {
                'min-max': 'minmax',
                'mean-plusminus-1-sigma': 1.,
                'mean-plusminus-2-sigma': 2.,
                'mean-plusminus-4-sigma': 4.,
        }

        if self._last_mode != mode:
            mi, ma = trace.minmax(
                self.traces, key=lambda tr: None, mode=modemap[mode])[None]
            self.ymin_data = mi
            self.ymax_data = ma
            self._last_mode = mode

        return self.ymin_data, self.ymax_data

    def set_yrange(self, ymin, ymax):
        if ymax == ymin:
            ymax = 0.5*(ymin + ymax) + 1.0
            ymin = 0.5*(ymin + ymax) - 1.0

        self.ymin = ymin
        self.ymax = ymax

    @property
    def tyrange(self):
        return (self.tmin, self.tmax), (self.ymin, self.ymax)

    def draw(self, plotenv, markers):
        self._draw_traces(plotenv)
        self._draw_time_label(plotenv)
        self._draw_markers(plotenv, markers)

    def _draw_time_label(self, plotenv):
        text = util.time_to_str(self.tmin, format=time_fmt_drumline(self.tmin))
        font = plotenv.style.label_textstyle.qt_font
        lab = Label(
            font.pointSize(), plotenv.vmin, text,
            anchor='ML', style=plotenv.style.label_textstyle)

        lab.draw(plotenv)

    def _draw_markers(self, plotenv, markers):
        project = plotenv.projector(((self.tmin, self.tmax), (-1., 1.)))
        plotenv.setPen(plotenv.style.marker_color.qt_color)

        rect = qc.QRectF(
            plotenv.umin, 0.,
            plotenv.umax-plotenv.umin, plotenv.widget.height())

        for marker in markers:
            plotenv.draw_vline(project, marker.tmin, -1., 1.)
            s = marker.get_label()
            if s:
                if isinstance(marker, PhaseMarker):
                    anchor = 'BC'
                    v = project.v(-1.)
                else:
                    anchor = 'BR'
                    v = project.v(-1.)

                lab = Label(
                    project.u(marker.tmin), v, s,
                    anchor=anchor,
                    style=plotenv.style.marker_textstyle,
                    keep_inside=rect,
                    head=(1.0, 3.0))

                lab.draw(plotenv)

    def _draw_traces(self, plotenv):
        project = plotenv.projector(self.tyrange)
        tpp = (project.xmax - project.xmin) / (project.umax - project.umin)
        if self._time_per_pixel != tpp:
            self._empty_cache()
            self._time_per_pixel = tpp

        for tr in self.traces:
            udata, vdata = self._projected_trace_data(
                tr,
                project,
                plotenv.style.trace_resolution)

            qpoints = make_QPolygonF(udata, vdata)
            plotenv.setPen(plotenv.style.trace_color.qt_color)
            plotenv.drawPolyline(qpoints)

    def _projected_trace_data(self, tr, project, trace_resolution):
        n = tr.data_len()
        if trace_resolution > 0 \
                and n > 2 \
                and tr.deltat < 0.5 / trace_resolution*self._time_per_pixel:

            spp = int(self._time_per_pixel / tr.deltat / trace_resolution)
            if tr not in self._ydata_cache:
                nok = (tr.data_len() // spp) * spp
                ydata_rs = tr.ydata[:nok].reshape((-1, spp))
                ydata = num.empty((nok // spp)*2)
                ydata[::2] = num.min(ydata_rs, axis=1)
                ydata[1::2] = num.max(ydata_rs, axis=1)
                self._ydata_cache[tr] = ydata
            else:
                ydata = self._ydata_cache[tr]

            udata_min = float(
                project.u(tr.tmin))
            udata_max = float(
                project.u(tr.tmin+0.5*tr.deltat*spp*(ydata.size-1)))
        else:
            ydata = tr.ydata
            udata_min = float(project.u(tr.tmin))
            udata_max = float(project.u(tr.tmin+tr.deltat*(n-1)))

        vdata = project.v(ydata)
        udata = num.linspace(udata_min, udata_max, vdata.size)
        return udata, vdata

    def _empty_cache(self):
        self._ydata_cache = {}


def tlen(x):
    return x.tmax-x.tmin


def is_relevant(x, tmin, tmax):
    return tmax >= x.tmin and x.tmax >= tmin


class MarkerStore(object):
    def __init__(self):
        self.empty()

    def empty(self):
        self._by_tmin = pile.Sorted([], 'tmin')
        self._by_tmax = pile.Sorted([], 'tmax')
        self._by_tlen = pile.Sorted([], tlen)
        self._adjust_minmax()
        self._listeners = []

    def relevant(self, tmin, tmax, selector=None):
        if not self._by_tmin or not is_relevant(self, tmin, tmax):
            return []

        if selector is None:
            return [
                x for x in self._by_tmin.with_key_in(tmin-self.tlenmax, tmax)
                if is_relevant(x, tmin, tmax)]
        else:
            return [
                x for x in self._by_tmin.with_key_in(tmin-self.tlenmax, tmax)
                if is_relevant(x, tmin, tmax) and selector(x)]

    def insert(self, x):
        self._by_tmin.insert(x)
        self._by_tmax.insert(x)
        self._by_tlen.insert(x)

        self._adjust_minmax()
        self._notify_listeners('insert', x)

    def remove(self, x):
        self._by_tmin.remove(x)
        self._by_tmax.remove(x)
        self._by_tlen.remove(x)

        self._adjust_minmax()
        self._notify_listeners('remove', x)

    def insert_many(self, x):
        self._by_tmin.insert_many(x)
        self._by_tmax.insert_many(x)
        self._by_tlen.insert_many(x)

        self._adjust_minmax()
        self._notify_listeners('insert_many', x)

    def remove_many(self, x):
        self._by_tmin.remove_many(x)
        self._by_tmax.remove_many(x)
        self._by_tlen.remove_many(x)

        self._adjust_minmax()
        self._notify_listeners('remove_many', x)

    def add_listener(self, obj):
        self._listeners.append(weakref.ref(obj))

    def _adjust_minmax(self):
        if self._by_tmin:
            self.tmin = self._by_tmin.min().tmin
            self.tmax = self._by_tmax.max().tmax
            self.tlenmax = tlen(self._by_tlen.max())
        else:
            self.tmin = None
            self.tmax = None
            self.tlenmax = None

    def _notify_listeners(self, what, x):
        for ref in self._listeners:
            obj = ref()
            if obj:
                obj(what, x)

    def __iter__(self):
        return iter(self._by_tmin)


class DrumViewMain(qw.QWidget, Listener):

    def __init__(self, pile, *args):
        qw.QWidget.__init__(self, *args)

        self.setAttribute(qc.Qt.WA_AcceptTouchEvents, True)

        st = self.state = State()
        self.markers = MarkerStore()
        self.markers.add_listener(self.listener_no_args(self._markers_changed))

        self.pile = pile
        self.pile.add_listener(self.listener(self._pile_changed))

        self._drumlines = {}
        self._wheel_pos = 0
        self._iline_float = None
        self._project_iline_to_screen = ProjectYV(
            (st.iline-0.5, st.iline+st.nlines+0.5), (0., self.height()))
        self._access_counter = 0
        self._waiting_for_first_data = True

        self._init_touch()
        self._init_following()

        sal = self.state.add_listener

        sal(self.listener_no_args(self._state_changed))
        sal(self.listener_no_args(self._drop_cached_drumlines), 'filters')
        sal(self.listener_no_args(self._drop_cached_drumlines), 'nslc')
        sal(self.listener_no_args(self._drop_cached_drumlines), 'tline')
        sal(self.listener_no_args(
            self._adjust_background_color), 'style.background_color')
        sal(self.listener_no_args(self._adjust_follow), 'follow')

        self._adjust_background_color()
        self._adjust_follow()

    def goto_data_begin(self):
        if self.pile.tmin:
            self.state.tmin = self.pile.tmin

    def goto_data_end(self):
        if self.pile.tmax:
            self.state.tmax = self.pile.tmax

    def next_nslc(self):
        nslc_ids = sorted(self.pile.nslc_ids.keys())
        if nslc_ids:
            try:
                i = nslc_ids.index(self.state.nslc)
            except ValueError:
                i = -1

            self.state.nslc = nslc_ids[(i+1) % len(nslc_ids)]

    def title(self):
        return ' '.join(x for x in self.state.nslc if x)

    def _state_changed(self):
        self.update()

    def _markers_changed(self):
        self.update()

    def _pile_changed(self, what, content):
        self._adjust_first_data()

        delete = []
        tline = self.state.tline
        for iline in self._drumlines.keys():
            for c in content:
                if c.overlaps(iline*tline, (iline+1)*tline):
                    delete.append(iline)

        for iline in delete:
            del self._drumlines[iline]

        self.update()

    def _init_following(self):
        self._follow_timer = None
        self._following_interrupted = False
        self._following_interrupted_tstart = None

    def interrupt_following(self):
        self._following_interrupted = True
        self._following_interrupted_tstart = time.time()

    def _follow_update(self):
        if self._following_interrupted:
            now = time.time()
            if self._following_interrupted_tstart < now - 20.:
                self._following_interrupted = False
            else:
                return

        now = time.time()
        iline = int(math.ceil(now / self.state.tline))-self.state.nlines
        if iline != self.state.iline:
            self.state.iline = iline

    def _adjust_follow(self):
        follow = self.state.follow
        if follow and not self._follow_timer:
            self._follow_timer = qc.QTimer(self)
            self._follow_timer.timeout.connect(self._follow_update)
            self._follow_timer.setInterval(1000)
            self._follow_update()
            self._follow_timer.start()

        elif not follow and self._follow_timer:
            self._follow_timer.stop()

    def _draw(self, plotenv):
        self._adjust_first_data()
        plotenv.umin = 0.
        plotenv.umax = self.width()
        self._draw_title(plotenv)
        self._draw_time_axis(plotenv)
        self._draw_lines(plotenv)

    def _draw_title(self, plotenv):
        font = plotenv.style.title_textstyle.qt_font
        lab = Label(
            0.5*(plotenv.umin + plotenv.umax),
            font.pointSize(),
            self.title(),
            anchor='TC',
            style=plotenv.style.title_textstyle)

        lab.draw(plotenv)

    def _draw_time_axis(self, plotenv):
        pass

    def _draw_lines(self, plotenv):
        st = self.state
        drumlines_seen = []
        for iline in range(st.iline, st.iline+st.nlines):
            self._update_line(iline)
            drumline = self._drumlines.get(iline, None)
            if drumline:
                drumlines_seen.append(drumline)

        self._autoscale(drumlines_seen)

        top_margin = 50.
        bottom_margin = 50.

        self._project_iline_to_screen = ProjectYV(
                (st.iline-0.5, st.iline+st.nlines-0.5),
                (top_margin, self.height()-bottom_margin))

        for drumline in drumlines_seen:
            plotenv.vmin = self._project_iline_to_screen.v(drumline.iline-0.5)
            plotenv.vmax = self._project_iline_to_screen.v(drumline.iline+0.5)
            markers = self._relevant_markers(drumline.tmin, drumline.tmax)
            drumline.draw(plotenv, markers)
            drumline.access_counter = self._access_counter
            self._access_counter += 1

        drumlines_by_access = sorted(
            self._drumlines.values(), key=lambda dl: dl.access_counter)

        for drumline in drumlines_by_access[:-st.npages_cache*st.nlines]:
            del self._drumlines[drumline.iline]

    def _relevant_markers(self, tmin, tmax):
        return self.markers.relevant(
            tmin, tmax,
            lambda m: not m.nslc_ids or m.match_nslc(self.state.nslc))

    def _autoscale(self, drumlines):
        if not drumlines:
            return

        st = self.state

        data = []
        for drumline in drumlines:
            try:
                data.append(drumline.data_range(st.scaling.base))
            except Empty:
                pass

        if not data:
            data = [[0, 0]]

        mi, ma = num.array(data, dtype=float).T
        gain = st.scaling.gain
        if st.scaling.mode == 'same':
            ymin, ymax = mi.min(), ma.max()
            for drumline in drumlines:
                drumline.set_yrange(ymin/gain, ymax/gain)
        elif st.scaling.mode == 'individual':
            for drumline, ymin, ymax in zip(drumlines, mi, ma):
                drumline.set_yrange(ymin/gain, ymax/gain)
        elif st.scaling.mode == 'fixed':
            for drumline in drumlines:
                drumline.set_yrange(st.scaling.min/gain, st.scaling.max/gain)

    def _update_line(self, iline):

        if iline not in self._drumlines:
            st = self.state
            tmin = iline*st.tline
            tmax = (iline+1)*st.tline
            if st.filters:
                tpad = max(x.tpad() for x in st.filters)
            else:
                tpad = 0.0

            traces = self.pile.all(
                    tmin=iline*st.tline,
                    tmax=(iline+1)*st.tline,
                    tpad=tpad,
                    trace_selector=lambda tr: tr.nslc_id == st.nslc,
                    keep_current_files_open=True,
                    accessor_id=id(self))

            for tr in traces:
                for filter in st.filters:
                    filter.apply(tr)

            self._drumlines[iline] = DrumLine(
                iline, tmin, tmax, traces, self.state)

    def _drop_cached_drumlines(self):
        self._drumlines = {}

    def _adjust_background_color(self):
        color = self.state.style.background_color.qt_color

        p = qg.QPalette()
        p.setColor(qg.QPalette.Background, color)
        self.setAutoFillBackground(True)
        self.setPalette(p)

    def _adjust_first_data(self):
        if self._waiting_for_first_data:
            if self.pile.tmin:
                self.next_nslc()
                self.goto_data_end()
                self._waiting_for_first_data = False

    # qt event handlers

    def paintEvent(self, event):
        plotenv = PlotEnv(self)
        plotenv.style = self.state.style
        plotenv.widget = self

        if plotenv.style.antialiasing:
            plotenv.setRenderHint(qg.QPainter.Antialiasing)

        self._draw(plotenv)

    def event(self, event):

        if event.type() in (
                qc.QEvent.TouchBegin,
                qc.QEvent.TouchUpdate,
                qc.QEvent.TouchEnd,
                qc.QEvent.TouchCancel):

            return self._touch_event(event)

        return qw.QWidget.event(self, event)

    def _init_touch(self):
        self._gesture = None

    def _touch_event(self, event):
        if event.type() == qc.QEvent.TouchBegin:
            self._gesture = DrumGesture(self)
            self._gesture.update(event)

        elif event.type() == qc.QEvent.TouchUpdate:
            self._gesture.update(event)

        elif event.type() == qc.QEvent.TouchEnd:
            self._gesture.update(event)
            self._gesture.end()
            self._gesture = None

        elif event.type() == qc.QEvent.TouchCancel:
            self._gesture.update(event)
            self._gesture.cancel()
            self._gesture = None

        return True

    def wheelEvent(self, event):

        self.interrupt_following()

        self._wheel_pos += event.angleDelta().y()

        n = self._wheel_pos // 120
        self._wheel_pos = self._wheel_pos % 120
        if n == 0:
            return

        amount = max(1., self.state.nlines/24.)
        wdelta = amount * n

        if event.modifiers() & qc.Qt.ControlModifier:
            proj = self._project_iline_to_screen

            anchor = (
                proj.y(event.y()) - proj.ymin) / (proj.ymax - proj.ymin)

            nlines = max(1, self.state.nlines + int(round(wdelta)))

            if self._iline_float is None:
                iline_float = float(self.state.iline)
            else:
                iline_float = self._iline_float

            self._iline_float = iline_float-anchor*wdelta

            self.state.iline = int(round(iline_float))
            self.state.nlines = nlines

        else:
            self.state.iline -= int(wdelta)
            self._iline_float = None

    def keyPressEvent(self, event):

        keytext = str(event.text())

        if event.key() == qc.Qt.Key_PageDown:
            self.interrupt_following()
            self.state.iline += self.state.nlines

        elif event.key() == qc.Qt.Key_PageUp:
            self.interrupt_following()
            self.state.iline -= self.state.nlines

        elif keytext == '+':
            self.state.scaling.gain *= 1.5

        elif keytext == '-':
            self.state.scaling.gain *= 1.0 / 1.5

        elif keytext == ' ':
            self.next_nslc()

        elif keytext == 'p':
            print(self.state)


tline_choices = [10., 30., 60., 120., 300., 600., 1200., 3600.]


class DrumGesture(object):

    def __init__(self, view):
        self._view = view
        self._active = False
        self.begin()

    def begin(self):
        self._active = True
        self._iline = self._view.state.iline
        self._nlines = self._view.state.nlines
        self._tline = self._view.state.tline
        self._proj = self._view._project_iline_to_screen
        self._gain = self._view.state.scaling.gain
        self._max_len_tps = 0

    def end(self):
        self._active = False

    def cancel(self):
        self.end()

    def update(self, event):
        if not self._active:
            return

        tps = event.touchPoints()
        proj = self._proj

        self._max_len_tps = max(len(tps), self._max_len_tps)

        if len(tps) == 1 and self._max_len_tps < 2:
            tp = tps[0]
            iline = proj.y(tp.pos().y())
            iline_start = proj.y(tp.startPos().y())
            idelta = int(round(iline - iline_start))
            iline = self._iline - idelta
            if iline != self._view.state.iline:
                self._view.interrupt_following()
                self._view.state.iline = iline

        if len(tps) == 2:
            # self._view.state.iline = self._iline

            u = [
                (tp.pos().x(), tp.startPos().x())
                for tp in tps]

            y = [
                (proj.y(tp.pos().y()), proj.y(tp.startPos().y()))
                for tp in tps]

            v = [
                (tp.pos().y(), tp.startPos().y())
                for tp in tps]

            if abs(u[0][1] - u[1][1]) < abs(v[0][1] - v[1][1]):

                vclip = self._view.size().height() * 0.05
                d0 = max(vclip, abs(v[0][1] - v[1][1]))
                d1 = max(vclip, abs(v[0][0] - v[1][0]))

                nlines = min(max(1, int(round(self._nlines * (d0 / d1)))), 50)

                idelta = int(round(
                    0.5 * ((y[0][0]+y[1][0]) - (y[0][1]+y[1][1]))
                    * (nlines / self._nlines)))

                iline = self._iline - idelta

                self._view.interrupt_following
                if self._view.state.nlines != nlines \
                        or self._view.state.iline != iline:
                    self._view.interrupt_following()
                    self._view.state.iline = iline
                    self._view.state.nlines = nlines

            else:
                if abs(u[0][0] - u[0][1]) + abs(u[1][0] - u[1][1]) \
                        > abs(v[0][0] - v[0][1]) + abs(v[1][0] - v[1][1]):

                    ustretch = abs(u[0][1] - u[1][1]) \
                        / (abs(u[0][0] - u[1][0])
                           + self._view.size().width() * 0.05)

                    print('ustretch', ustretch)
                    log_tline_choices = num.log(tline_choices)
                    log_tline = num.log(ustretch * self._tline)

                    ichoice = num.argmin(
                        num.abs(log_tline_choices - log_tline))
                    tline = tline_choices[ichoice]
                    # yanchor = 0.5 * (y[0][1] + y[1][1])
                    # r = (yanchor - proj.ymin) / (proj.ymax - proj.ymin)

                    if self._view.state.tline != tline:
                        self._view.state.iline = int(
                            round(self._iline / (tline / self._tline)))
                        self._view.state.tline = tline

                else:
                    vstretch = 10**(
                        -((v[0][0] - v[0][1]) + (v[1][0] - v[1][1]))
                        / self._view.size().height())

                    self._view.state.scaling.gain = self._gain * vstretch
