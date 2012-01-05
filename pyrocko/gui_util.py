import math, calendar, time
import numpy as num

import pyrocko.util, pyrocko.plot, pyrocko.model, pyrocko.trace
from pyrocko.util import TableWriter, TableReader
from pyrocko.nano import Nano

from PyQt4.QtCore import *
from PyQt4.QtGui import *

def gmtime_x(timestamp):
    if isinstance(timestamp, Nano):
        etimestamp = int(timestamp)
    else:
        etimestamp = math.floor(timestamp)
    tt = time.gmtime(etimestamp)
    ms = (timestamp-etimestamp)*1000
    return tt,ms
        
def mystrftime(fmt=None, tt=None, milliseconds=0):
   
    if fmt is None: fmt = '%Y-%m-%d %H:%M:%S .%r'
    if tt is None: tt = time.time()
    
    fmt2 = fmt.replace('%r', '%03i' % int(round(milliseconds)))
    fmt3 = fmt2.replace('%u', '%06i' % int(round(milliseconds*1000)))
    return time.strftime(fmt3, tt)
        
def myctime(timestamp):
    tt, ms = gmtime_x(timestamp)
    return mystrftime(None, tt, ms)

def str_to_float_or_none(s):
    if s == 'None':
        return None
    else:
        return float(s)

def str_to_str_or_none(s):
    if s == 'None':
        return None
    else:
        return s

def str_to_bool(s):
    return s.lower() in ('true', 't', '1')


def make_QPolygonF( xdata, ydata ):
    if isinstance(xdata, Nano):
        xdata = xdata.float_array()
        
    assert len(xdata) == len(ydata)
    qpoints = QPolygonF( len(ydata) )
    vptr = qpoints.data()
    vptr.setsize(len(ydata)*8*2)
    aa = num.ndarray( shape=(len(ydata),2), dtype=num.float64, buffer=buffer(vptr))
    aa.setflags(write=True)
    aa[:,0] = xdata
    aa[:,1] = ydata
    return qpoints

def draw_label( p, x,y, label_str, label_bg, anchor='BL', outline=False):
    fm = p.fontMetrics()
    
    label = QString( label_str )
    rect = fm.boundingRect( label )
    
    tx,ty =x,y
    if 'T' in anchor:
        ty += rect.height()
    if 'R' in anchor:
        tx -= rect.width()
    if 'M' in anchor:
        ty += rect.height()/2.
    if 'C' in anchor:
        tx -= rect.width()/2.
        
    rect.translate( tx, ty )
    if outline:
        oldpen = p.pen()
        oldbrush = p.brush()
        p.setBrush(label_bg)
        rect.adjust(-2.,0.,2.,0.)
        p.drawRect( rect )
        p.setPen(oldpen)
        p.setBrush(oldbrush)
        
    else:
        p.fillRect(rect, label_bg)
    p.drawText( tx, ty, label )

def get_err_palette():
    err_palette = QPalette()
    err_palette.setColor( QPalette.Base, QColor(255,200,200) )
    return err_palette

class MySlider(QSlider):
    
    def wheelEvent(self, ev):
        ev.ignore()

    def keyPressEvent(self, ev):
        ev.ignore()

class MyValueEdit(QLineEdit):

    def __init__(self, parent, low_is_none=False, high_is_none=False, *args):
        QLineEdit.__init__(self, *args)
        self.value = 0.
        self.mi = 0.
        self.ma = 1.
        self.low_is_none = low_is_none
        self.high_is_none = high_is_none
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
            t = str(self.text()).strip()
            if self.low_is_none and t in ('off', 'below'):
                value = self.mi
            elif self.high_is_none and t in ('off', 'above'):
                value = self.ma
            else:
                value = float(t)

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
        t = ('%8.5g' % self.value).strip()
        if self.low_is_none and self.value == self.mi:
            if self.high_is_none:
                t = 'below'
            else:
                t = 'off'

        if self.high_is_none and self.value == self.ma:
            if self.low_is_none:
                t = 'above'
            else:
                t = 'off'

        self.setText(t)
        
class ValControl(QFrame):

    def __init__(self, low_is_none=False, high_is_none=False, *args):
        apply(QFrame.__init__, (self,) + args)
        self.layout = QHBoxLayout( self )
        self.layout.setMargin(0)
        self.lname = QLabel( "name", self )
        self.lname.setMinimumWidth(120)
        self.lvalue = MyValueEdit( self, low_is_none=low_is_none, high_is_none=high_is_none )
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
        return int(round(math.log(value/self.mi) / a))
    
    def setup(self, name, mi, ma, cur, ind):
        self.lname.setText( name )
        self.mi = mi
        self.ma = ma
        self.ind = ind
        self.lvalue.setRange( self.s2v(0), self.s2v(10000) )
        self.set_value(cur)
        
    def set_value(self, cur):
        if cur is None:
            if self.low_is_none:
                cur = self.mi
            elif self.high_is_none:
                cur = self.ma

        self.mute = True
        self.cur = cur
        self.cursl = self.v2s(cur)
        self.slider.blockSignals(True)
        self.slider.setValue( self.cursl )
        self.slider.blockSignals(False)
        self.lvalue.blockSignals(True)
        if self.cursl in (0, 10000):
            self.lvalue.setValue( self.s2v(self.cursl) )
        else:
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
                self.cursl = cursl
            
            self.fire_valchange()
        
    def fire_valchange(self):
        if self.mute: return
        
        cur = self.cur

        if self.low_is_none and self.cursl == 0:
            cur = None

        if self.high_is_none and self.cursl == 10000:
            cur = None

        self.emit(SIGNAL("valchange(PyQt_PyObject,int)"), cur, int(self.ind) )
        
class LinValControl(ValControl):
    
    def s2v(self, svalue):
        return svalue/10000. * (self.ma-self.mi) + self.mi
                
    def v2s(self, value):
        return int(round((value-self.mi)/(self.ma-self.mi) * 10000.))

class MarkerParseError(Exception):
    pass

class MarkerOneNSLCRequired(Exception):
    pass

class Marker(object):
    
    @staticmethod
    def from_string(line):
        
        def fail():
            raise MarkerParseError(
                'Unable to create marker from string: "%s"' % line)
                
        def parsedate(ymd,hms,sfs):
            return calendar.timegm(time.strptime(ymd+' '+hms, '%Y-%m-%d %H:%M:%S')) + float(sfs)
        
        try:
            toks = line.split()
            if len(toks) in (4,5):
                tmin = parsedate(*toks[:3])
                tmax = tmin
                
            elif len(toks) in (8,9):
                tmin = parsedate(*toks[:3])
                tmax = parsedate(*toks[3:6])
            
            else:
                fail()
            
            if len(toks) in (5,9):
                kind = int(toks[-2])
            else:
                kind = int(toks[-1])
                
            nslc_ids = []
            if len(toks) in (5,9):
                for snslc in toks[-1].split(','):
                    nslc = snslc.split('.')
                    if len(nslc) != 4:
                        fail()
                        
                    nslc_ids.append(tuple(nslc))
                
        except MarkerParseError:
            fail()
    
        return Marker(nslc_ids, tmin, tmax, kind=kind)

    @staticmethod
    def save_markers(markers, fn):

        f = open(fn,'w')
        f.write('# Snuffler Markers File Version 0.2\n')
        writer = TableWriter(f)
        for marker in markers:
            a = marker.get_attributes()
            w = marker.get_attribute_widths()
            row = []
            for x in a:
                if x is None or x == '':
                    row.append('None')
                else:
                    row.append(x)

            writer.writerow(row, w)

        f.close()

    @staticmethod
    def load_markers(fn):
        markers = []
        f = open(fn, 'r')
        line  = f.readline()
        if not line.startswith('# Snuffler Markers File Version'):
            f.seek(0)
            for iline, line in enumerate(f):
                sline = line.strip()
                if not sline or sline.startswith('#'):
                    continue
                try:
                    m = Marker.from_string(sline)
                    markers.append(m)
                    
                except MarkerParseError:
                    logger.warn('Invalid marker definition in line %i of file "%s"' % (iline+1, fn))
                    
            f.close()
        
        elif line.startswith('# Snuffler Markers File Version 0.2'):
            reader = TableReader(f)
            while not reader.eof:
                row = reader.readrow()
                if not row:
                    continue
                if row[0] == 'event:':
                    marker = EventMarker.from_attributes(row)
                elif row[0] == 'phase:':
                    marker = PhaseMarker.from_attributes(row)
                else:
                    marker = Marker.from_attributes(row)
                 
                markers.append(marker)
        else:
            logger.warn('Unsupported Markers File Version')
        
        return markers

    def __init__(self, nslc_ids, tmin, tmax, kind=0):
        self.set(nslc_ids, tmin, tmax)
        c = pyrocko.plot.color
        self.color_a = [ c(x) for x in ('aluminium4', 'aluminium5', 'aluminium6') ]
        self.color_b = [ c(x) for x in ('scarletred1', 'scarletred2', 'scarletred3',
                                        'chameleon1', 'chameleon2', 'chameleon3',
                                        'skyblue1', 'skyblue2', 'skyblue3',
                                        'orange1', 'orange2', 'orange3',
                                        'plum1', 'plum2', 'plum3',
                                        'chocolate1', 'chocolate2', 'chocolate3') ]
        self.alerted = False
        self.selected = False
        self.kind = kind
        
    def set(self, nslc_ids, tmin,tmax):
        self.nslc_ids = nslc_ids
        self.tmin = tmin
        self.tmax = tmax
     
    def set_kind(self, kind):
        self.kind = kind
     
    def get_tmin(self):
        return self.tmin
        
    def get_tmax(self):
        return self.tmax

    def get_nslc_ids(self):
        return self.nslc_ids

    def is_alerted(self):
        return self.alerted
        
    def is_selected(self):
        return self.selected

    def set_alerted(self, state):
        self.alerted = state
        
    def set_selected(self, state):
        self.selected = state

    def match_nsl(self, nsl):
        patterns = [ '.'.join(x[:3]) for x in self.nslc_ids ]
        return pyrocko.util.match_nslc(patterns, nsl)
    
    def match_nslc(self, nslc):
        patterns = [ '.'.join(x) for x in self.nslc_ids ]
        return pyrocko.util.match_nslc(patterns, nslc)

    def one_nslc(self):
        if len(self.nslc_ids) != 1:
            raise MarkerOneNSLCRequired()

        return list(self.nslc_ids)[0]

    def __str__(self):
        traces = ','.join( [ '.'.join(nslc_id) for nslc_id in self.nslc_ids ] )
        st = myctime
        if self.tmin == self.tmax:
            return '%s %i %s' % (st(self.tmin), self.kind, traces)
        else:
            return '%s %s %g %i %s' % (st(self.tmin), st(self.tmax), self.tmax-self.tmin, self.kind, traces)

    def get_attributes(self):
        traces = ','.join( [ '.'.join(nslc_id) for nslc_id in self.nslc_ids ] )
        st = pyrocko.util.time_to_str
        vals = []
        vals.extend(st(self.tmin).split())
        if self.tmin != self.tmax:    
            vals.extend(st(self.tmax).split())
            vals.append(self.tmax-self.tmin)

        vals.append(self.kind)
        vals.append(traces)
        return vals

    def get_attribute_widths(self):
        ws = [ 10, 12 ]
        if self.tmin != self.tmax:
            ws.extend( [ 10, 12, 12 ] )
        ws.extend( [ 2, 15 ] )
        return ws

    @staticmethod
    def parse_attributes(vals):
        tmin = pyrocko.util.str_to_time( vals[0] + ' ' + vals[1] )
        i = 2
        tmax = tmin
        if len(vals) == 7:
            tmax = pyrocko.util.str_to_time( vals[2] + ' ' + vals[3] )
            i = 5

        kind = int(vals[i])
        traces = vals[i+1]
        if traces == 'None':
            nslc_ids = []
        else:
            nslc_ids = tuple([ tuple(nslc_id.split('.')) for nslc_id in traces.split(',') ])

        return nslc_ids, tmin, tmax, kind

    @staticmethod
    def from_attributes(vals):
        return Marker(*Marker.parse_attributes(vals))

    def select_color(self, colorlist):
        cl = lambda x: colorlist[(self.kind*3+x)%len(colorlist)]
        if self.selected:
            return cl(1)
        if self.alerted:
            return cl(1)
        return cl(2)
            
    def draw(self, p, time_projection, y_projection, draw_line=True, draw_triangle=False):
        
        if self.selected or self.alerted or not self.nslc_ids:
            
            color = self.select_color(self.color_b)            
            pen = QPen(QColor(*color))
            pen.setWidth(2)
            linepen = QPen(pen)
            if self.selected or self.alerted:
                linepen.setStyle(Qt.CustomDashLine)
                pat = [5.,3.]
                linepen.setDashPattern(pat)
                if self.alerted and not self.selected:
                    linepen.setColor(QColor(150,150,150))
            
            s = 9.
            utriangle = make_QPolygonF( [ -0.577*s, 0., 0.577*s ], [ 0., 1.*s, 0.] ) 
            ltriangle = make_QPolygonF( [ -0.577*s, 0., 0.577*s ], [ 0., -1.*s, 0.] ) 

            def drawline(t):
                u = time_projection(t)
                v0, v1 = y_projection.get_out_range()
                line = QLineF(u,v0,u,v1)
                p.drawLine(line)
        
            def drawtriangles(t):
                u = time_projection(t)
                v0, v1 = y_projection.get_out_range()
                t = QPolygonF(utriangle)
                t.translate(u,v0)
                p.drawConvexPolygon(t) 
                t = QPolygonF(ltriangle)
                t.translate(u,v1)
                p.drawConvexPolygon(t) 
           
            if draw_line or self.selected or self.alerted:
                p.setPen(linepen)
                drawline(self.tmin)
                drawline(self.tmax)

            if draw_triangle:
                pen.setStyle(Qt.SolidLine)
                pen.setJoinStyle(Qt.MiterJoin)
                pen.setWidth(2)
                p.setPen(pen)
                p.setBrush(QColor(*color))
                drawtriangles(self.tmin)


    def draw_trace(self, p, trace, time_projection, track_projection, gain, outline_label=False):
        if self.nslc_ids and not self.match_nslc(trace.nslc_id): return
        
        color = self.select_color(self.color_b)
        pen = QPen(QColor(*color))
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        def drawpoint(t,y):
            u = time_projection(t)
            v = track_projection(y)
            rect = QRectF(u-2,v-2,4,4)
            p.drawRect(rect)
            
        def drawline(t):
            u = time_projection(t)
            v0, v1 = track_projection.get_out_range()
            line = QLineF(u,v0,u,v1)
            p.drawLine(line)

        try:
            snippet = trace.chop(self.tmin, self.tmax, inplace=False, include_last=True, snap=(math.ceil,math.floor))
            
            vdata = track_projection( gain*snippet.get_ydata() )
            udata_min = float(time_projection(snippet.tmin))
            udata_max = float(time_projection(snippet.tmin+snippet.deltat*(vdata.size-1)))
            udata = num.linspace(udata_min, udata_max, vdata.size)
            qpoints = make_QPolygonF( udata, vdata )
            pen.setWidth(1)
            p.setPen(pen)
            p.drawPolyline( qpoints )
            pen.setWidth(2)
            p.setPen(pen)
            drawpoint(*trace(self.tmin, clip=True, snap=math.ceil))
            drawpoint(*trace(self.tmax, clip=True, snap=math.floor))
            
        except pyrocko.trace.NoData:
            pass
            
        color = self.select_color(self.color_b)
        pen = QPen(QColor(*color))
        pen.setWidth(2)
        p.setPen(pen)

        drawline(self.tmin)
        drawline(self.tmax)

        label = self.get_label()
        if label:
            label_bg = QBrush( QColor(255,255,255) )
            
            u = time_projection(self.tmin)
            v0, v1 = track_projection.get_out_range()
            if outline_label:
                du = -7
            else:
                du = -5
            draw_label( p, u+du, v0, label, label_bg, 'TR', outline=outline_label)

        if self.tmin == self.tmax:
            try: drawpoint(self.tmin, trace.interpolate(self.tmin))
            except IndexError: pass
        #try: drawpoint(self.tmax, trace.interpolate(self.tmax))
        #except IndexError: pass            
    
    def get_label(self):
        return None

    def convert_to_phase_marker(self, event=None, phasename=None, polarity=None, automatic=None):
        if isinstance(self, PhaseMarker):
            return

        self.__class__ = PhaseMarker
        self._event = event
        self._phasename = phasename
        self._polarity = polarity
        self._automatic = automatic

    def convert_to_event_marker(self, lat=0., lon=0.):
        if isinstance(self, EventMarker):
            return

        if isinstance(self, PhaseMarker):
            self.convert_to_marker()

        self.__class__ = EventMarker
        self._event = pyrocko.model.Event( lat, lon, self.tmin, name='Event')
        self._active = False
        self.tmax = self.tmin
        self.nslc_ids = []

class EventMarker(Marker):
    def __init__(self, event, kind=0, event_hash=None):
        Marker.__init__(self, [], event.time, event.time, kind)
        self._event = event
        self._active = False
        self._event_hash = event_hash

    def get_event_hash(self):
        if self._event_hash is not None:
            return self._event_hash
        else:
            self._event.get_hash()

    def set_active(self, active):
        self._active = active

    def label(self):
        t = []
        mag = self._event.magnitude
        if mag is not None:
            t.append('M%3.1f' % mag)
        
        reg = self._event.region
        if reg is not None:
            t.append(reg)
       
        nam = self._event.name
        if nam is not None:
            t.append(nam)

        return ' '.join(t)

    def draw(self, p, time_projection, y_projection):
      
        Marker.draw(self, p, time_projection, y_projection, draw_line=False, draw_triangle=True)
        
        u = time_projection(self.tmin)
        v0, v1 = y_projection.get_out_range()
        label_bg = QBrush( QColor(255,255,255) )
        draw_label( p, u, v0-10., self.label(), label_bg, 'CB', outline=self._active)

    def get_event(self):
        return self._event

    def draw_trace(self, p, trace, time_projection, track_projection, gain):
        pass
    
    def get_attributes(self):
        attributes = [ 'event:' ]
        attributes.extend(Marker.get_attributes(self))
        del attributes[-1]
        e = self._event
        attributes.extend([e.get_hash(), e.lat, e.lon, e.depth, e.magnitude, e.catalog, e.name, e.region ])
        return attributes

    def get_attribute_widths(self):
        ws = [ 6 ]
        ws.extend( Marker.get_attribute_widths(self) )
        del ws[-1]
        ws.extend([ 14, 12, 12, 12, 4, 5, 0, 0 ])
        return ws

    @staticmethod
    def from_attributes(vals):

        nslc_ids, tmin, tmax, kind = Marker.parse_attributes(vals[1:] + [ 'None' ])
        lat, lon, depth, magnitude = [ str_to_float_or_none( x ) for x in vals[5:9] ]
        catalog, name, region = [ str_to_str_or_none(x) for x in vals[9:] ]
        e = pyrocko.model.Event(lat, lon, tmin, name, depth, magnitude, region, catalog=catalog)
        marker = EventMarker(e, kind, event_hash=str_to_str_or_none(vals[4]))
        return marker

class PhaseMarker(Marker):

    def __init__(self, nslc_ids, tmin, tmax, kind, event=None, event_hash=None, phasename=None, polarity=None, automatic=None):
        Marker.__init__(self, nslc_ids, tmin, tmax, kind)
        self._event = event
        self._event_hash = event_hash
        self._phasename = phasename
        self._polarity = polarity
        self._automatic = automatic

    def draw_trace(self, p, trace, time_projection, track_projection, gain):
        Marker.draw_trace(self, p, trace, time_projection, track_projection, gain, outline_label=self._event is not None)
         
    def get_label(self):
        t = []
        if self._phasename is not None:
            t.append(self._phasename)
        if self._polarity is not None:
            t.append(self._polarity)

        if self._automatic:
            t.append('@')

        return ''.join(t)

    def get_event(self):
        return self._event

    def get_event_hash(self):
        if self._event_hash is not None:
            return self._event_hash
        else:
            return self._event.get_hash()

    def set_event_hash(self, event_hash):
        self._event_hash = event_hash

    def set_event(self, event):
        self._event = event

    def get_phasename(self):
        return self._phasename

    def set_phasename(self, phasename):
        self._phasename = phasename

    def convert_to_marker(self):
        del self._event
        del self._phasename
        del self._polarity
        del self._automatic
        self.__class__ = Marker

    def get_attributes(self):
        attributes = [ 'phase:' ]
        attributes.extend(Marker.get_attributes(self))
        h = None
        et = None, None
        if self._event:
            h = self._event.get_hash()
            et = pyrocko.util.time_to_str(self._event.time).split()

        attributes.extend([h, et[0], et[1], self._phasename, self._polarity, self._automatic])
        return attributes

    def get_attribute_widths(self):
        ws = [ 6 ]
        ws.extend( Marker.get_attribute_widths(self) )
        ws.extend([ 14, 12, 12, 8, 4, 5 ])
        return ws

    @staticmethod
    def from_attributes(vals):
        nslc_ids, tmin, tmax, kind = Marker.parse_attributes(vals[1:])
       
        i = 8
        if len(vals) == 14:
            i = 11
       
        event_hash = str_to_str_or_none( vals[i-3] )
        phasename, polarity = [ str_to_str_or_none( x ) for x in vals[i:i+2] ]
        automatic = str_to_bool( vals[i+2] )
        marker = PhaseMarker( nslc_ids, tmin, tmax, kind, event=None, event_hash=event_hash,
            phasename=phasename, polarity=polarity, automatic=automatic )
        return marker
