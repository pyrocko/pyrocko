'''Snuffling infrastructure

This module provides the base class 'Snuffling' for user-defined snufflings and 
some utilities for their handling.
'''


import os, sys, logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import pile

from gui_util import ValControl

logger = logging.getLogger('pyrocko.snufflings')

class Param:
    '''Definition of an adjustable parameter for the snuffling. The snuffling
    may display controls for user input for such parameters.'''
    
    def __init__(self, name, ident, default, minimum, maximum):
        self.name = name
        self.ident = ident
        self.default = default
        self.minimum = minimum
        self.maximum = maximum

class SnufflingModule:
    '''Utility class to load/reload snufflings from a file.
    
    The snufflings are created by a user module which has a special method
    __snufflings__() which return the snuffling instances to be exported. The
    snuffling module is attached to a handler class, which makes use of the
    snufflings (e.g. PileOverwiew from pile_viewer.py). The handler class must
    implement the methods add_snuffling() and remove_snuffling() which are used
    as callbacks. The callbacks are utilized from the methods load_if_needed()
    and remove_snufflings() which are called from the handler class, when
    needed.
    '''
    
    mtimes = {}
    
    def __init__(self, path, name, handler):
        self._path = path
        self._name = name
        self._mtime = None
        self._module = None
        self._snufflings = []
        self._handler = handler
        
    def load_if_needed(self):
        filename = os.path.join(self._path, self._name+'.py')
        mtime = os.stat(filename)[8]
        sys.path[0:0] = [ self._path ]
        if self._module == None:
            self._module = __import__(self._name)
            for snuffling in self._module.__snufflings__():
                self.add_snuffling(snuffling)
            
        elif self._mtime != mtime:
            logger.warn('reloading snuffling module %s' % self._name)
            self.remove_snufflings()
            reload(self._module)
            for snuffling in self._module.__snufflings__():
                self.add_snuffling(snuffling)
            
        self._mtime = mtime
        sys.path[0:1] = []
    
    def add_snuffling(self, snuffling):
        self._snufflings.append(snuffling)
        self._handler.add_snuffling(snuffling)
    
    def remove_snufflings(self):
        for snuffling in self._snufflings:
            self._handler.remove_snuffling(snuffling)
            
        self._snufflings = []
    
    
class NoViewerSet(Exception):
    pass

class Snuffling:
    '''Base class for user snufflings.
    
    Snufflings are plugins for snuffler (and other applications using the
    PileOverview class from pile_viewer.py). They can be added, removed and
    reloaded at runtime and should provide a simple way of extending the
    functionality of snuffler.
    
    A snuffling has access to all data available in a pile viewer, can process 
    this data and can create and add new traces and markers to the viewer.
    '''

    def __init__(self):
        self._name = 'Untitled Snuffling'
        self._viewer = None
        self._tickets = []
        
        self._delete_panel = None
        self._delete_menuitem = None
        
        self._panel_parent = None
        self._panel_hook = None
        self._menu_parent = None
        self._menu_hook = None
        
        self._panel = None
        self._menuitem = None
        self._parameters = []
        
        self.setup()
        
    def setup(self):
        '''Setup the snuffling.
        
        This method should be implemented in subclass and contain e.g. calls to 
        set_name() and add_parameter().
        '''
        
        pass
    
    def init_gui(self, viewer, panel_parent, panel_hook, menu_parent, menu_hook):
        '''Set parent viewer and hooks to add panel and menu entry.
        
        This method is called from the PileOverview object. Calls setup_gui().
        '''
        
        self._viewer = viewer
        self._panel_parent = panel_parent
        self._panel_hook = panel_hook
        self._menu_parent = menu_parent
        self._menu_hook = menu_hook
        
        self.setup_gui()
        
    def setup_gui(self):
        '''Create and add gui elements to the viewer.
        
        This method is initially called from init_gui(). It is also called,
        e.g. when new parameters have been added or if the name of the snuffling 
        has been changed.
        '''
        
        self._panel = self.make_panel(self._panel_parent)
        if self._panel:
            self._delete_panel = self._panel_hook(self.get_name(), self._panel)
        
        self._menuitem = self.make_menuitem(self._menu_parent)
        if self._menuitem:
            self._delete_menuitem = self._menu_hook(self._menuitem)
        
    def delete_gui(self):
        '''Remove the gui elements of the snuffling.
        
        This removes the panel and menu entry of the widget from the viewer and
        also removes all traces added with the add_traces() method.
        '''
        
        self.cleanup()
        
        if self._delete_panel is not None:
            self._delete_panel()
            self._panel = None
            
        if self._delete_menuitem is not None:
            self._delete_menuitem()
            self._menuitem = None
            
    def set_name(self, name):
        '''Set the snuffling's name.
        
        The snuffling's name is shown as a menu entry and in the panel header.
        '''
        
        self._name = name
        
        if self._panel or self._menuitem:
            self.delete_gui()
            self.setup_gui()
    
    def get_name(self):
        '''Get the snuffling's name.'''
        
        return self._name
    
    def add_parameter(self, param):
        '''Add an adjustable parameter to the snuffling.
        
            param -- object of type Param
        
        For each parameter added, controls are added to the snuffling's panel,
        so that the parameter can be adjusted from the gui.
        '''
        
        self._parameters.append(param)
        self.set_parameter(param.ident, param.default)
        
        if self._panel is not None:
            self.delete_gui()
            self.setup_gui()
    
    def get_parameters(self):
        '''Get the snufflings adjustable parameter definitions.
        
        
        Returns a list of objects of type Param.
        '''
        
        return self._parameters
    
    def set_parameter(self, ident, value):
        '''Set one of the snuffling's adjustable parameters.
        
            ident -- identifier of the parameter
            value -- new value of the parameter
            
        This is usually called when the parameter has been set via the gui 
        controls.
        '''
        
        setattr(self, ident, value)
        
    def get_viewer(self):
        '''Get the parent viewer.
        
        Returns a reference to an object of type PileOverview, which is the
        main viewer widget.
        
        If no gui has been initialized for the snuffling, a NoViewerSet 
        exception is raised.
        '''
        
        if self._viewer is None:
            raise NoViewerSet()
        return self._viewer
    
    def get_pile(self):
        '''Get the pile.
        
        If a gui has been initialized, a reference to the viewer's internal pile
        is returned. If not, the make_pile() method (which may be overloaded in
        subclass) is called to create a pile. This can be utilized to make 
        hybrid snufflings, which may work also in a standalone mode.
        '''
        
        try:
            p =self.get_viewer().get_pile()
        except NoViewerSet:
            p = self.make_pile()
            
        return p
        
    def make_pile(self):
        '''Create a pile.
        
        To be overloaded in subclass. The default implementation just calls
        pyrocko.pile.make_pile() to create a pile from command line arguments.
        '''
        
        pile.make_pile()
        
    def make_panel(self, parent):
        '''Create a widget for the snuffling's control panel.
        
        Normally called from the setup_gui() method. Returns None if no panel
        is needed (e.g. if the snuffling has no adjustable parameters).'''
    
        params = self.get_parameters()
        
        if params:                
            frame = QFrame(parent)
            layout = QGridLayout()
            frame.setLayout( layout )
                        
            for iparam, param in enumerate(params):
                param_widget = ValControl()
                param_widget.setup(param.name, param.minimum, param.maximum, param.default, iparam)
                self.get_viewer().connect( param_widget, SIGNAL("valchange(float,int)"), self.modified_snuffling_panel )
                layout.addWidget( param_widget, iparam, 0 )
        
            return frame
            
        else:
            return None
    
    def make_menuitem(self, parent):
        '''Create the menu item for the snuffling.
        
        This method may be overloaded in subclass and return None, if no menu 
        entry is wanted.
        '''
        
        item = QAction(self.get_name(), parent)
        self.get_viewer().connect( item, SIGNAL("triggered(bool)"), self.menuitem_triggered )
        return item
    
    def modified_snuffling_panel(self, value, iparam):
        '''Called when the user has played with an adjustable parameter.
        
        The default implementation sets the parameter, calls the snuffling's 
        call() method and finally triggers an update on the viewer widget.
        '''
        
        param = self.get_parameters()[iparam]
        self.set_parameter(param.ident, value)
        self.call()
        self.get_viewer().update()
        
    def menuitem_triggered(self, arg):
        '''Called when the user has triggered the snuffling's menu.
        
        The default implementation calls the snuffling's call() method and triggers
        an update on the viewer widget.'''
        
        self.call()
        self.get_viewer().update().
        
    def add_traces(self, traces):
        '''Add traces to the viewer.
        
            traces -- list of objects of type pyrocko.trace.Trace
            
        The traces are put into a MemTracesFile and added to the viewer's 
        internal pile for display. Note, that unlike with the traces from the 
        files given on the command line, these traces are kept in memory and so
        may quickly occupy a lot of ram if a lot of traces are added.
        
        This method should be preferred over modifying the viewer's internal 
        pile directly, because this way, the snuffling has a chance to
        automatically remove its private traces again (see cleanup() method).
        '''
        
        ticket = self.get_viewer().add_traces(traces)
        self._tickets.append( ticket )
        return ticket

    def cleanup(self):
        '''Remove all traces which have been added so far by the snuffling.'''
        
        self.get_viewer().release_data(self._tickets)
        self._tickets = []
    
    def call(self):
        '''Main work routine of the snuffling.
        
        This method is called when the snuffling's menu item has been triggered
        or when the user has played with the panel controls. To be overloaded in
        subclass. The default implementation does nothing useful.
        '''
        
        pass
    
    def __del__(self):
        self.cleanup()

        
