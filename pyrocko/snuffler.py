#!/usr/bin/env python

'''Effective seismological trace viewer.'''

import os, sys, signal, logging, time, re, gc
import numpy as num

import pyrocko.pile
import pyrocko.util
import pyrocko.pile_viewer
import pyrocko.model

from PyQt4.QtCore import *
from PyQt4.QtGui import *

logger = logging.getLogger('pyrocko.snuffler')

class MyMainWindow(QMainWindow):

    def __init__(self, app, *args):
        QMainWindow.__init__(self, *args)
        self.app = app

    def keyPressEvent(self, ev):
        self.app.pile_viewer.get_view().keyPressEvent(ev)

class Snuffler(QApplication):
    
    def __init__(self):
        QApplication.__init__(self, [])
        self.connect(self, SIGNAL("lastWindowClosed()"), self.myQuit)
        signal.signal(signal.SIGINT, self.myCloseAllWindows)

    def snuffle(self,*args, **kwargs):
        self.win = SnufflerWindow(*args, **kwargs)
    
    def myCloseAllWindows(self, *args):
        self.closeAllWindows()
    
    def myQuit(self, *args):
        self.quit()

    def return_tag(self):
        return self.win.return_tag()

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
            self.pile_viewer.get_view().add_stations(stations)
       
        if events:
            for ev in events:
                self.pile_viewer.get_view().add_event(ev)
            
            self.pile_viewer.get_view().set_origin(events[0])

        if markers:
            self.pile_viewer.get_view().add_markers(markers)

        
        self.tabs = SnufflerTabs(self)
        self.setCentralWidget( self.tabs )
        self.add_tab('Main', self.pile_viewer)

        self.pile_viewer.setup_snufflings()

        self.add_panel('Main Controls', self.pile_viewer.controls(), visible=controls)
        self.show()

        self.pile_viewer.get_view().setFocus(Qt.OtherFocusReason)

        sb = self.statusBar()
        sb.clearMessage()
        sb.showMessage('Welcome to Snuffler! Click and drag to zoom and pan. Doubleclick to pick. Right-click for Menu. <space> to step forward. <b> to step backward. <q> to close.')
            
        if follow:
            self.pile_viewer.get_view().follow(float(follow))

    def dockwidgets(self):
        return [ w for w in self.findChildren(QDockWidget) if not w.isFloating() ]

    def get_panel_parent_widget(self):
        return self

    def add_tab(self, name, widget):
        self.tabs.append_tab(widget, name)

    def add_panel(self, name, panel, visible=False):
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
        self.pile_viewer.get_view().add_panel_toggler(mitem)

        self.dockwidget_to_toggler[dockwidget] = mitem


    def toggle_panel(self, dockwidget, visible):
        dockwidget.setVisible(visible)
        if visible:
            dockwidget.setFocus()
            dockwidget.raise_()

    def remove_panel(self, panel):
        dockwidget = panel.parent()
        self.removeDockWidget(dockwidget)
        dockwidget.setParent(None)
        mitem = self.dockwidget_to_toggler[dockwidget]
        self.pile_viewer.get_view().remove_panel_toggler(mitem)
        
    def return_tag(self):
        return self.pile_viewer.get_view().return_tag
    
app = None

def snuffle(pile, **kwargs):
    '''View pile in a snuffler window.
    
    :param pile: :py:class:`pyrocko.pile.Pile` object to be visualized
    :param stations: list of `pyrocko.model.Station` objects or ``None``
    :param events: list of `pyrocko.model.Event` objects or ``None``
    :param markers: list of `pyrocko.gui_util.Marker` objects or ``None``
    :param ntracks: float, number of tracks to be shown initially (default: 12)
    :param follow: time interval (in seconds) for real time follow mode or ``None``
    :param controls: bool, whether to show the main controls (default: ``True``)
    :param opengl: bool, whether to use opengl (default: ``False``)
    '''
    
    global app
    if app is None:
        app = Snuffler()

    app.snuffle(pile, **kwargs)
    app.exec_()
    return app.return_tag()


