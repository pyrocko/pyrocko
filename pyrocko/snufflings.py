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
    
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.module = None
        
    def load(self):
        filename = os.path.join(self.path, self.name+'.py')
        mtime = os.stat(filename)[8]
        sys.path[0:0] = [ self.path ]
        self.module = __import__(self.name)
        if filename in SnufflingModule.mtimes:
            if SnufflingModule.mtimes[filename] != mtime:
                logger.warn('reloading snuffling module %s' % self.name)
                reload(self.module)
        SnufflingModule.mtimes[filename] = mtime
        sys.path[0:1] = []
    
    def snufflings(self):
        self.load()
        return self.module.__snufflings__()

class NoViewerSet(Exception):
    pass

class Snuffling:

    def __init__(self):
        self._viewer = None
        self._tickets = []
        for param in self.get_parameters():
            self.set_parameter(param.ident, param.default)
        
        self._delete_panel = None
    
    
    def init_gui(self, panel_parent, panel_hook):
        panel = self.make_panel(panel_parent)
        if panel:
            self._delete_panel = panel_hook(self.get_name(), panel)
        
    def delete_gui(self):
        self.cleanup()
        if self._delete_panel is not None:
            self._delete_panel()
    
    def set_viewer(self, viewer):
        self._viewer = viewer
        
    def get_viewer(self):
        if self._viewer is None:
            raise NoViewerSet()
        return self._viewer
        
        
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
                layout.addWidget( param_widget, iparam,0 )
        
            return frame
            
        else:
            return None
    
    def modified_snuffling_panel(self, value, iparam):
        param = self.get_parameters()[iparam]
        self.set_parameter(param.ident, value)
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

        
