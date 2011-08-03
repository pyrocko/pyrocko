import math
import numpy as num


from PyQt4.QtCore import *
from PyQt4.QtGui import *

def get_err_palette():
    err_palette = QPalette()
    err_palette.setColor( QPalette.Base, QColor(255,200,200) )
    return err_palette

class MySlider(QSlider):
    
    def wheelEvent(self, ev):
        ev.ignore()

class MyValueEdit(QLineEdit):

    def __init__(self, *args):
        apply(QLineEdit.__init__, (self,) + args)
        self.value = 0.
        self.mi = 0.
        self.ma = 1.
        self.connect( self, SIGNAL("editingFinished()"), self.myEditingFinished )
        self.lock = False
        
    def setRange( self, mi, ma ):
        self.mi = mi
        self.ma = ma
        
    def setValue( self, value ):
        if not self.lock:
            self.value = value
            self.setPalette( QApplication.palette() )
            self.adjust_text()
        
    def myEditingFinished(self):
        try:
            value = float(str(self.text()).strip())
            if not (self.mi <= value <= self.ma):
                raise Exception("out of range")
            if value != self.value:
                self.value = value
                self.lock = True
                self.emit(SIGNAL("edited(float)"), value )
                self.setPalette( QApplication.palette() )
        except:
            self.setPalette( get_err_palette() )
        
        self.lock = False
        
    def adjust_text(self):
        self.setText( ("%8.5g" % self.value).strip() )
        
class ValControl(QFrame):

    def __init__(self, low_is_none=False, high_is_none=False, *args):
        apply(QFrame.__init__, (self,) + args)
        self.layout = QHBoxLayout( self )
        self.layout.setMargin(0)
        #self.layout.setSpacing(5)
        self.lname = QLabel( "name", self )
        self.lname.setMinimumWidth(120)
        self.lvalue = MyValueEdit( self )
        self.lvalue.setFixedWidth(100)
        self.slider = MySlider(Qt.Horizontal, self)
        self.slider.setMaximum( 10000 )
        self.slider.setSingleStep( 100 )
        self.slider.setPageStep( 1000 )
        self.slider.setTickPosition( QSlider.NoTicks )
        self.slider.sizePolicy().setHorizontalStretch(10)
        self.slider.setFocusPolicy(Qt.ClickFocus)
        self.layout.addWidget( self.lname )
        self.layout.addWidget( self.lvalue )
        self.layout.addWidget( self.slider )
        self.low_is_none = low_is_none
        self.high_is_none = high_is_none
        #self.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.connect( self.slider, SIGNAL("valueChanged(int)"),
                      self.slided )
        self.connect( self.lvalue, SIGNAL("edited(float)"),
                      self.edited )
        self.mute = False
    
    def s2v(self, svalue):
        a = math.log(self.ma/self.mi) / 10000.
        return self.mi*math.exp(a*svalue)
                
    def v2s(self, value):
        a = math.log(self.ma/self.mi) / 10000.
        return math.log(value/self.mi) / a
    
    def setup(self, name, mi, ma, cur, ind):
        self.lname.setText( name )
        self.mi = mi
        self.ma = ma
        self.ind = ind
        self.lvalue.setRange( mi, ma )
        self.set_value(cur)
        
    def set_value(self, cur):
        self.mute = True
        self.cur = cur
        self.cursl = self.v2s(cur)
        self.slider.blockSignals(True)
        self.slider.setValue( self.cursl )
        self.slider.blockSignals(False)
        self.lvalue.blockSignals(True)
        self.lvalue.setValue( self.cur )
        self.lvalue.blockSignals(False)
        self.mute = False
        
    def get_value(self):
        return self.cur
        
    def slided(self,val):
        if self.cursl != val:
            self.cursl = val
            self.cur = self.s2v(self.cursl)

            self.lvalue.blockSignals(True)
            self.lvalue.setValue( self.cur )
            self.lvalue.blockSignals(False)
            self.fire_valchange()
            
    def edited(self,val):
        if self.cur != val:
            self.cur = val
            cursl = self.v2s(val)
            if (cursl != self.cursl):
                self.slider.blockSignals(True)
                self.slider.setValue( cursl )
                self.slider.blockSignals(False)
            
            self.fire_valchange()
        
    def fire_valchange(self):
        if self.mute: return
        
        if self.low_is_none and self.cursl == 0:
            cur = num.nan
        else:
            cur = self.cur
            
        if self.high_is_none and self.cursl == 10000:
            cur = num.nan
        else:
            cur = self.cur
                        
        self.emit(SIGNAL("valchange(float,int)"), cur, int(self.ind) )
        
class LinValControl(ValControl):
    
    def s2v(self, svalue):
        return svalue/10000. * (self.ma-self.mi) + self.mi
                
    def v2s(self, value):
        return (value-self.mi)/(self.ma-self.mi) * 10000.

