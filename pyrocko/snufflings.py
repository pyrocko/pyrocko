import os, sys, logging

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from gui_util import ValControl

logger = logging.getLogger('pyrocko.snufflings')

class Param:
    def __init__(self, name, ident, default, minimum, maximum):
        self.name = name
        self.ident = ident
        self.default = default
        self.minimum = minimum
        self.maximum = maximum

class SnufflingModule:
    
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

    def __init__(self):
        self._viewer = None
        self._tickets = []
        for param in self.get_parameters():
            self.set_parameter(param.ident, param.default)
        
        self._delete_panel = None
        self._delete_menuitem = None
    
    def init_gui(self, panel_parent, panel_hook, menu_parent, menu_hook):
        panel = self.make_panel(panel_parent)
        if panel:
            self._delete_panel = panel_hook(self.get_name(), panel)
        
        menuitem = self.make_menuitem(menu_parent)
        if menuitem:
            self._delete_menuitem = menu_hook(menuitem)
        
    def delete_gui(self):
        self.cleanup()
        if self._delete_panel is not None:
            self._delete_panel()
        if self._delete_menuitem is not None:
            self._delete_menuitem()
            
    def set_viewer(self, viewer):
        self._viewer = viewer
        
    def get_viewer(self):
        if self._viewer is None:
            raise NoViewerSet()
        return self._viewer
    
    def get_pile(self):
        return self.get_viewer().get_pile()
    
    def get_parameters(self):
        return []
    
    def set_parameter(self, ident, value):
        setattr(self, ident, value)
        
    def make_panel(self, parent):
    
        params = self.get_parameters()
        
        if params:                
            frame = QFrame(parent)
            layout = QGridLayout()
            frame.setLayout( layout )
            #layout.setRowStretch(0,1)
                        
            for iparam, param in enumerate(params):
                param_widget = ValControl()
                param_widget.setup(param.name, param.minimum, param.maximum, param.default, iparam)
                self.get_viewer().connect( param_widget, SIGNAL("valchange(float,int)"), self.modified_snuffling_panel )
                layout.addWidget( param_widget, iparam, 0 )
        
            return frame
            
        else:
            return None
    
    def make_menuitem(self, parent):
        item = QAction(self.get_name(), parent)
        self.get_viewer().connect( item, SIGNAL("triggered(bool)"), self.menuitem_triggered )
        return item
    
    def modified_snuffling_panel(self, value, iparam):
        param = self.get_parameters()[iparam]
        self.set_parameter(param.ident, value)
        self.call()
        self.get_viewer().update()
        
    def menuitem_triggered(self, arg):
        self.call()
        self.get_viewer().update()
        
    def add_traces(self, traces):
        ticket = self.get_viewer().add_traces(traces)
        self._tickets.append( ticket )
        return ticket

    def cleanup(self):
        self.get_viewer().release_data(self._tickets)
        self._tickets = []
            
    def __del__(self):
        self.cleanup()

        
