import os, sys

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from gui_util import ValControl

class Param:
    def __init__(self, name, ident, minimum, maximum, default):
        self.name = name
        self.ident = ident
        self.minimum = minimum
        self.maximum = maximum
        self.default = default

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
        
    def set_viewer(self, viewer):
        self._viewer = viewer
        
    def get_viewer(self):
        if self._viewer is None:
            raise NoViewerSet()
        return self._viewer
        
    def panel(self, parent):
    
        params = self.get_parameters()
        
        if params:                
            frame = QFrame(parent)
            layout = QGridLayout()
            frame.setLayout( layout )
            layout.setRowStretch(0,1)
                        
            for iparam, param in enumerate(self.get_parameters()):
                param_widget = ValControl()
                param_widget.setup(param.name, param.minimum, param.maximum, param.default, iparam)
                #self.connect( param_widget, SIGNAL("valchange(float,int)"), self.modified_snuffling_panel )
                layout.addWidget( param_widget, iparam,0 )
        
            return frame
            
        else:
            return None
    
    def add_traces(self, traces):
        ticket = self.get_viewer().add_traces(traces)
        self._tickets.append( ticket )
        return ticket

    def cleanup(self):
        self.get_viewer().release_data(self._tickets)
        self._tickets = []
            
    def __del__(self):
        self.cleanup()

        
