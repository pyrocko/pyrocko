
import numpy as num

from PyQt4.QtCore import *
from PyQt4.QtGui import *

import mopad

from pile_viewer import make_QPolygonF, ValControl, LinValControl, Projection


class BeachballView(QWidget):
    
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        mt = mopad.MomentTensor(M=(1,2,3,4,5,6))
        self.set_moment_tensor(mt)
        
    def set_moment_tensor(self, mt):
        self.beachball = mopad.BeachBall(MT=mt)
        self.beachball._setup_BB()
        self.update()
        
    def paintEvent(self, paint_ev ):
        '''Called by QT whenever widget needs to be painted.'''
        
        painter = QPainter(self)
        painter.setRenderHint( QPainter.Antialiasing )
        self.drawit( painter )
        
        
    def drawit(self, p):
        '''Draw beachball into painter.'''
       
        h = self.height()
        w = self.width()

        s = min(h,w)*0.9
        
        xproj = Projection()
        xproj.set_in_range(-1.,1.)
        xproj.set_out_range((w-s)/2.,w-(w-s)/2.)
        
        yproj = Projection()
        yproj.set_in_range(-1.,1.)
        yproj.set_out_range(h-(h-s)/2.,(h-s)/2.)
        
       
        pos, neg = self.beachball.get_projected_nodallines()
        
        color = (0,0,0)
        pen = QPen(QColor(*color))
        pen.setWidth(2)
        p.setPen(pen)

        for line in pos, neg:
            x = xproj(num.array(line[0]))
            y = yproj(num.array(line[1]))
            
            points = make_QPolygonF(x,y)
            p.drawPolyline( points )
            
        p.drawEllipse(QRectF(QPointF(xproj(-1.), yproj(-1.)), QPointF(xproj(1.), yproj(1.))))
        
        
class MomentTensorEditor(QFrame):
    
    def __init__(self, *args):
        QFrame.__init__(self, *args)
        
        self.moment_tensor = mopad.MomentTensor(M=(1,2,3,4,5,6))
        
        setupdata = [
            (LinValControl, 'Strike 1', 0., 360., 0., 0),
            (LinValControl, 'Dip 1', 0., 90., 0., 1),
            (LinValControl, 'Slip-Rake 1', -180., 180., 0., 2),
            (LinValControl, 'Strike 2', 0., 360., 0., 3),
            (LinValControl, 'Dip 2', 0., 90., 0., 4),
            (LinValControl, 'Slip-Rake 2', -180., 180., 0., 5)]
        
        layout = QGridLayout()
        self.setLayout( layout )
        
        widgets = []
        for i, (typ, name, vmin, vmax, vcur, ind) in enumerate(setupdata):
            widget = typ()
            widget.setup(name, vmin, vmax, vcur, ind)
            widgets.append(widget)
            layout.addWidget(widget, i, 0)
            self.connect( widget, SIGNAL('valchange(float,int)'), self.valchange )
            
        self.widgets = widgets
        self.adjust_values()
        
            
    def adjust_values(self):
        
        (strike1, dip1,rake1), (strike2,dip2,rake2) = self.moment_tensor.get_fps()
        
        for widget, value in zip(self.widgets, [strike1, dip1, rake1, strike2, dip2, rake2]):
            widget.set_value(value)
        
    def valchange(self, val, ind):
        strike, dip, rake = [ widget.get_value() for widget in self.widgets[:3] ]
        
        self.moment_tensor = mopad.MomentTensor(M=(strike,dip,rake))
        self.adjust_values()
            
        self.emit(SIGNAL('moment_tensor_changed(PyQt_PyObject)'), self.moment_tensor )
