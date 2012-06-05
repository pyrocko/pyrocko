#!/usr/bin/env python

'''Effective seismological trace viewer.'''

import os, sys, signal, logging, time, re, gc
from optparse import OptionParser
import numpy as num

import pyrocko.pile
import pyrocko.util
import pyrocko.pile_viewer
import pyrocko.model
import pyrocko.config

from PyQt4.QtCore import *
from PyQt4.QtGui import *

logger = logging.getLogger('pyrocko.snuffler')

class Connection(QObject):
    def __init__(self, parent, sock):
        QObject.__init__(self, parent)
        self.socket = sock
        self.connect(sock, SIGNAL('readyRead()'), self.handle_read)
        self.connect(sock, SIGNAL('disconnected()'), self.handle_disconnected)
        self.nwanted = 8
        self.reading_size = True
        self.handler = None
        self.nbytes_received = 0
        self.nbytes_sent = 0
        self.compressor = zlib.compressobj()
        self.decompressor = zlib.decompressobj()

    def handle_read(self):
        while True:
            navail = self.socket.bytesAvailable()
            if navail < self.nwanted:
                return

            data = self.socket.read(self.nwanted)
            self.nbytes_received += len(data)
            if self.reading_size:
                self.nwanted = struct.unpack('>Q', data)[0]
                self.reading_size = False
            else:
                obj = pickle.loads(self.decompressor.decompress(data))
                if obj is None:
                    self.socket.disconnectFromHost()
                else:
                    self.handle_received(obj)
                self.nwanted = 8
                self.reading_size = True

    def handle_received(self, obj):
        self.emit(SIGNAL('received(PyQt_PyObject,PyQt_PyObject)'), self, obj)

    def ship(self, obj):
        data = self.compressor.compress(pickle.dumps(obj))
        data_end = self.compressor.flush(zlib.Z_FULL_FLUSH)
        self.socket.write(struct.pack('>Q', len(data)+len(data_end)))
        self.socket.write(data)
        self.socket.write(data_end)
        self.nbytes_sent += len(data)+len(data_end) + 8

    def handle_disconnected(self):
        self.emit(SIGNAL('disconnected(PyQt_PyObject)'), self)

    def close(self):
        self.socket.close()

class ConnectionHandler(QObject):
    def __init__(self, parent):
        QObject.__init__(self, parent)
        self.queue = []
        self.connection = None

    def connected(self):
        return self.connection == None

    def set_connection(self, connection):
        self.connection = connection
        self.connect(connection, SIGNAL('received(PyQt_PyObject,PyQt_PyObject)'), self._handle_received)
        self.connect(connection, SIGNAL('disconnected(PyQt_PyObject)'), self.handle_disconnected)
        for obj in self.queue:
            self.connection.ship(obj)
        self.queue = []

    def _handle_received(self, conn, obj):
        self.handle_received(obj)

    def handle_received(self, obj):
        pass

    def handle_disconnected(self):
        self.connection = None

    def ship(self, obj):
        if self.connection:
            self.connection.ship(obj)
        else:
            self.queue.append(obj)

class SimpleConnectionHandler(ConnectionHandler):
    def __init__(self, parent, **mapping):
        ConnectionHandler.__init__(self, parent)
        self.mapping = mapping

    def handle_received(self, obj):
        command = obj[0]
        args = obj[1:]
        self.mapping[command](*args)


class MyMainWindow(QMainWindow):

    def __init__(self, app, *args):
        QMainWindow.__init__(self, *args)
        self.app = app

    def keyPressEvent(self, ev):
        self.app.pile_viewer.get_view().keyPressEvent(ev)


class SnufflerTabs(QTabWidget):
    def __init__(self, parent):
        QTabWidget.__init__(self, parent)
        if hasattr(self, 'setTabsClosable'):
            self.setTabsClosable(True)
        self.connect(self, SIGNAL('tabCloseRequested(int)'), self.removeTab)
        if hasattr(self, 'setDocumentMode'):
            self.setDocumentMode(True)

    def hide_close_button_on_first_tab(self):
        tbar = self.tabBar()
        if hasattr(tbar ,'setTabButton'):
            tbar.setTabButton(0, QTabBar.LeftSide, None)
            tbar.setTabButton(0, QTabBar.RightSide, None)

    def append_tab(self, widget, name):
        widget.setParent(self)
        self.insertTab(self.count(), widget, name)
        self.setCurrentIndex(self.count()-1)

    def tabInserted(self, index):
        if index == 0:
            self.hide_close_button_on_first_tab()

        self.tabbar_visibility()

    def tabRemoved(self, index):
        self.tabbar_visibility()

    def tabbar_visibility(self):
        if self.count() <= 1:
            self.tabBar().hide()
        elif self.count() > 1:
            self.tabBar().show()

class SnufflerWindow(QMainWindow):

    def __init__(self, pile, stations=None, events=None, markers=None, 
                        ntracks=12, follow=None, controls=True, opengl=False):
        
        QMainWindow.__init__(self)

        self.dockwidget_to_toggler = {}
            
        self.setWindowTitle( "Snuffler" )        

        self.pile_viewer = pyrocko.pile_viewer.PileViewer(
            pile, ntracks_shown_max=ntracks, use_opengl=opengl, panel_parent=self)
       
        if stations:
            self.get_view().add_stations(stations)
       
        if events:
            for ev in events:
                self.get_view().add_event(ev)
            
            self.get_view().set_origin(events[0])

        if markers:
            self.get_view().add_markers(markers)

        
        self.tabs = SnufflerTabs(self)
        self.setCentralWidget( self.tabs )
        self.add_tab('Main', self.pile_viewer)

        self.pile_viewer.setup_snufflings()

        self.add_panel('Main Controls', self.pile_viewer.controls(), visible=controls)
        self.show()

        self.get_view().setFocus(Qt.OtherFocusReason)

        sb = self.statusBar()
        sb.clearMessage()
        sb.showMessage('Welcome to Snuffler! Click and drag to zoom and pan. Doubleclick to pick. Right-click for Menu. <space> to step forward. <b> to step backward. <q> to close.')

        if follow:
            self.get_view().follow(float(follow))
        
        self.closing = False
    
    def sizeHint(self):
        return QSize(1024,768)

    def keyPressEvent(self, ev):
        self.get_view().keyPressEvent(ev)

    def get_view(self):
        return self.pile_viewer.get_view()

    def dockwidgets(self):
        return [ w for w in self.findChildren(QDockWidget) if not w.isFloating() ]

    def get_panel_parent_widget(self):
        return self

    def add_tab(self, name, widget):
        self.tabs.append_tab(widget, name)

    def add_panel(self, name, panel, visible=False, volatile=False):
        dws = self.dockwidgets()
        dockwidget = QDockWidget(name, self)
        dockwidget.setWidget(panel)
        panel.setParent(dockwidget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dockwidget)

        if dws:
            self.tabifyDockWidget(dws[-1], dockwidget)
        
        self.toggle_panel(dockwidget, visible)

        mitem = QAction(name, None)
        
        def toggle_panel(checked):
            self.toggle_panel(dockwidget, True)

        self.connect( mitem, SIGNAL('triggered(bool)'), toggle_panel)

        if volatile:
            def visibility(visible):
                if not visible:
                    self.remove_panel(panel)

            self.connect( dockwidget, SIGNAL('visibilityChanged(bool)'), visibility)

        self.get_view().add_panel_toggler(mitem)
        self.dockwidget_to_toggler[dockwidget] = mitem

    def toggle_panel(self, dockwidget, visible):
        dockwidget.setVisible(visible)
        if visible:
            w = dockwidget.widget()
            minsize = w.minimumSize()
            w.setMinimumHeight( w.sizeHint().height()+5 )
            def reset_minimum_size():
                w.setMinimumSize( minsize )
            
            QTimer.singleShot( 200, reset_minimum_size )

            dockwidget.setFocus()
            dockwidget.raise_()

    def remove_panel(self, panel):
        dockwidget = panel.parent()
        self.removeDockWidget(dockwidget)
        dockwidget.setParent(None)
        mitem = self.dockwidget_to_toggler[dockwidget]
        self.get_view().remove_panel_toggler(mitem)
        
    def return_tag(self):
        return self.get_view().return_tag

    def closeEvent(self, event):
        event.accept()
        self.closing = True

    def is_closing(self):
        return self.closing
    
class Snuffler(QApplication):
    
    def __init__(self):
        QApplication.__init__(self, sys.argv)
        self.connect(self, SIGNAL("lastWindowClosed()"), self.myQuit)
        signal.signal(signal.SIGINT, self.myCloseAllWindows)
        self.server = None
        self.loader = None

    def start_server(self):
        self.connections = []
        s = QTcpServer(self)
        s.listen(QHostAddress.LocalHost)
        self.connect(s, SIGNAL('newConnection()'), self.handle_accept)
        self.server = s

    def start_loader(self):
        self.loader = SimpleConnectionHandler(self, add_files=self.add_files, update_progress=self.update_progress)
        ticket = os.urandom(32)
        self.forker.spawn('loader', self.server.serverPort(), ticket)
        self.connection_handlers[ticket] = self.loader

    def handle_accept(self):
        sock = self.server.nextPendingConnection()
        con = Connection(self, sock)
        self.connections.append(con)
        self.connect(con, SIGNAL('disconnected(PyQt_PyObject)'), self.handle_disconnected) 
        self.connect(con, SIGNAL('received(PyQt_PyObject,PyQt_PyObject)'), self.handle_received_ticket)

    def handle_disconnected(self, connection):
        self.connections.remove(connection)
        connection.close()
        del connection

    def handle_received_ticket(self, connection, object):
        if not isinstance(object, str):
            self.handle_disconnected(connection)

        ticket = object
        if ticket in self.connection_handlers:
            h = self.connection_handlers[ticket]
            self.disconnect(connection, SIGNAL('received(PyQt_PyObject,PyQt_PyObject)'), self.handle_received_ticket)
            h.set_connection(connection)
        else:
            self.handle_disconnected(connection)

    def snuffler_windows(self):
        return [ w for w in self.topLevelWidgets() 
                    if isinstance(w, SnufflerWindow) and not w.is_closing() ]

    def event(self, e):
        if isinstance(e, QFileOpenEvent):
            paths = [ str(e.file()) ]
            wins = self.snuffler_windows()
            if wins:
                wins[0].get_view().load_soon(paths)

            return True
        else:
            return QApplication.event(self, e)

    def load(pathes, cachedirname, pattern, format):
        if not self.loader:
            self.start_loader()

        self.loader.ship(('load', args, self.cachedirname, options.pattern, options.format ))

    def add_files(self, files):
        p = self.pile_viewer.get_pile()
        p.add_files(files)
        self.pile_viewer.update_contents()

    def update_progress(self, task, percent):
        self.pile_viewer.progressbars.set_status(task, percent)

    def myCloseAllWindows(self, *args):
        self.closeAllWindows()
    
    def myQuit(self, *args):
        self.quit()

app = None

def snuffle(pile=None, **kwargs):
    '''View pile in a snuffler window.
    
    :param pile: :py:class:`pyrocko.pile.Pile` object to be visualized
    :param stations: list of `pyrocko.model.Station` objects or ``None``
    :param events: list of `pyrocko.model.Event` objects or ``None``
    :param markers: list of `pyrocko.gui_util.Marker` objects or ``None``
    :param ntracks: float, number of tracks to be shown initially (default: 12)
    :param follow: time interval (in seconds) for real time follow mode or ``None``
    :param controls: bool, whether to show the main controls (default: ``True``)
    :param opengl: bool, whether to use opengl (default: ``False``)
    :param paths: list of files and directories to search for trace files
    :param pattern: regex which filenames must match
    :param progressive: bool, whether to load files in progressive mode
    :param format: format of input files
    :param cache_dir: cache directory with trace meta information
    :param force_cache: bool, whether to use the cache when attribute spoofing is active
    '''
    
    if pile is None:
        pile = pyrocko.pile.make_pile()
    
    global app
    if app is None:
        app = Snuffler()
    
    kwargs_load = {}
    for k in ('paths', 'regex', 'progressive', 'format', 'cache_dir', 'force_cache'):
        try:
            kwargs_load[k] = kwargs.pop(k)
        except KeyError:
            pass

    win = SnufflerWindow(pile, **kwargs)
    
    if 'paths' in kwargs_load:
        win.get_view().load(**kwargs_load)

    if not win.is_closing():
        app.exec_()

    ret = win.return_tag()
    
    del win
    gc.collect()

    return ret

def snuffler_from_commandline(args=sys.argv):

    usage = '''usage: %prog [options] waveforms ...'''
    parser = OptionParser(usage=usage)

    parser.add_option('--format',
            dest='format',
            default='from_extension',
            choices=('mseed', 'sac', 'kan', 'segy', 
                'seisan', 'seisan_l', 'seisan_b', 'from_extension', 'try'),
            help='assume files are of given FORMAT [default: \'%default\']' )

    parser.add_option('--pattern',
            dest='regex',
            metavar='REGEX',
            help='only include files whose paths match REGEX')

    parser.add_option('--stations',
            dest='station_fns',
            action='append',
            default=[],
            metavar='STATIONS',
            help='read station information from file STATIONS')

    parser.add_option('--event', '--events',
            dest='event_fns',
            action='append',
            default=[],
            metavar='EVENT',
            help='read event information from file EVENT')

    parser.add_option('--markers',
            dest='marker_fns',
            action='append',
            default=[],
            metavar='MARKERS',
            help='read marker information file MARKERS')

    parser.add_option('--follow',
            dest='follow',
            metavar='N',
            help='follow real time with a window of N seconds')

    parser.add_option('--progressive',
            dest='progressive',
            action='store_true',
            default=False,
            help='don\'t wait for file scanning to complete before opening the viewer')
    
    parser.add_option('--cache',
            dest='cache_dir',
            default=pyrocko.config.cache_dir,
            metavar='DIR',
            help='use directory DIR to cache trace metadata (default=\'%default\')')

    parser.add_option('--force-cache',
            dest='force_cache',
            action='store_true',
            default=False,
            help='use the cache even when trace attribute spoofing is active (may have silly consequences)')

    parser.add_option('--ntracks',
            dest='ntracks',
            default=24,
            metavar='N',
            help='initially use N waveform tracks in viewer [default: %default]')

    parser.add_option('--opengl',
            dest='opengl',
            action='store_true',
            default=False,
            help='use OpenGL for drawing')

    parser.add_option('--debug',
            dest='debug',
            action='store_true',
            default=False,
            help='print debugging information to stderr')
    
    options, args = parser.parse_args(list(args[1:]))

    if options.debug:
        pyrocko.util.setup_logging('snuffler', 'debug')
    else:
        pyrocko.util.setup_logging('snuffler', 'warning')

    
    pile = pyrocko.pile.Pile()
    stations = []
    for stations_fn in options.station_fns:
        stations.extend(pyrocko.model.load_stations(stations_fn))
    
    events = []
    for event_fn in options.event_fns:
        events.extend(pyrocko.model.Event.load_catalog(event_fn))
    
    markers = []
    for marker_fn in options.marker_fns:
        markers.extend(pyrocko.pile_viewer.Marker.load_markers(marker_fn))
    
    return snuffle( pile,
            stations=stations,
            events=events,
            markers=markers,
            ntracks=options.ntracks,
            follow=options.follow,
            controls=True,
            opengl=options.opengl,
            paths=args, 
            progressive=options.progressive,
            cache_dir=options.cache_dir,
            regex=options.regex,
            format=options.format,
            force_cache=options.force_cache)



