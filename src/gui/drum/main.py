# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function, division

import sys
import tempfile
import os
import signal
import gc

from pyrocko.gui.qt_compat import qw
from pyrocko import pile, orthodrome, cake
from pyrocko.client import catalog
from pyrocko.gui.util import EventMarker, PhaseMarker

from pyrocko.gui.drum import view, state

pjoin = os.path.join


def event_markers(tmin, tmax, magmin=6.):
    cat = catalog.Geofon()
    event_names = cat.get_event_names(
        time_range=(tmin, tmax),
        magmin=magmin)

    markers = []
    for event_name in event_names:
        event = cat.get_event(event_name)
        markers.append(EventMarker(event))

    return markers


def phase_markers(
        events, stations, phase_names='Pg,P,PKP,PKIKP,S,PP,SS'.split(',')):

    mod = cake.load_model()
    phases = []
    for name in phase_names:
        for phase in cake.PhaseDef.classic(name):
            phase.name = name
            phases.append(phase)

    markers = []
    for event in events:
        for station in stations:
            dist = orthodrome.distance_accurate50m(event, station)
            depth = event.depth
            if depth is None:
                depth = 0.0

            rays = mod.arrivals(
                phases=phases,
                distances=[dist*cake.m2d],
                zstart=depth)

            for ray in rays:
                time = ray.t
                name = ray.given_phase().name
                incidence_angle = ray.incidence_angle()
                takeoff_angle = ray.takeoff_angle()

                time += event.time
                m = PhaseMarker(
                    [(station.network, station.station, '*', '*')],
                    time,
                    time,
                    2,
                    phasename=name,
                    event=event,
                    incidence_angle=incidence_angle,
                    takeoff_angle=takeoff_angle)

                markers.append(m)

    return markers


class App(qw.QApplication):
    def __init__(self):
        qw.QApplication.__init__(self, sys.argv)
        self.lastWindowClosed.connect(self.myQuit)
        self._main_window = None

    def install_sigint_handler(self):
        self._old_signal_handler = signal.signal(
            signal.SIGINT, self.myCloseAllWindows)

    def uninstall_sigint_handler(self):
        signal.signal(signal.SIGINT, self._old_signal_handler)

    def myQuit(self, *args):
        self.quit()

    def myCloseAllWindows(self, *args):
        self.closeAllWindows()

    def set_main_window(self, win):
        self._main_window = win

    def get_main_window(self):
        return self._main_window


app = None


def main(*args, **kwargs):

    from pyrocko import util
    from pyrocko.gui.snuffler.snuffler_app import PollInjector, \
        setup_acquisition_sources

    util.setup_logging('drumplot', 'info')

    global app
    global win

    p = pile.Pile()

    paths = ['school:///dev/ttyACM0?rate=40&gain=4&station=LEGO']
    store_path = 'recording'
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    store_interval = 600.

    paths.append(store_path)

    sources = setup_acquisition_sources(paths)
    if store_path is None:
        tempdir = tempfile.mkdtemp('', 'drumplot-tmp-')
        store_path = pjoin(
            tempdir,
            '%(network)s.%(station)s.%(location)s.%(channel)s.%(tmin)s.mseed')

    elif os.path.isdir(store_path):
        store_path = pjoin(
            store_path,
            '%(network)s.%(station)s.%(location)s.%(channel)s.%(tmin)s.mseed')

    if app is None:
        app = App()

    win = view.DrumViewMain(p)

    pollinjector = PollInjector(
        p,
        fixation_length=store_interval,
        path=store_path)

    for source in sources:
        source.start()
        pollinjector.add_source(source)

    fns = util.select_files(
        paths,
        selector=None,
        regex=None,
        show_progress=False)

    p.load_files(fns, show_progress=False)

    win.state.style.antialiasing = True
    win.state.tline = 600.
    # win.state.style.background_color = state.Color(r=0.2,g=0.27,b=0.36)
    # win.state.style.trace_color = state.Color(r=0.9,g=0.9,b=0.9)
    # win.state.style.title_textstyle.color = state.Color(r=1.0,g=1.0,b=1.0)
    # win.state.style.label_textstyle.color = state.Color(r=1.0,g=1.0,b=1.0)
    win.state.filters = [state.Demean()]
    win.state.scaling.min = -100.
    win.state.scaling.max = 100.
    win.state.scaling.mode = 'fixed'

    win.state.follow = True

    # pile_nsl = set(x[:3] for x in p.nslc_ids.keys())
    # stations = [
    #     s for s in model.load_stations(stations_fn) if s.nsl() in pile_nsl]

    # emarks = event_markers(p.tmin, p.tmax)
    # pmarks = phase_markers(
    #     events=[m.get_event() for m in emarks],
    #     stations=stations)
    #
    # win.markers.insert_many(emarks)
    # win.markers.insert_many(pmarks)

    win.show()

    app.set_main_window(win)

    app.install_sigint_handler()
    app.exec_()
    app.uninstall_sigint_handler()

    for source in sources:
        source.stop()

    if pollinjector:
        pollinjector.fixate_all()

    del win

    gc.collect()
