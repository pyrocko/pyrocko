import math, calendar, time
import numpy as num

import pyrocko.util, pyrocko.plot, pyrocko.model, pyrocko.trace, pyrocko.plot
from pyrocko.util import TableWriter, TableReader

from PyQt4.QtCore import *
from PyQt4.QtGui import *

def gmtime_x(timestamp):
    etimestamp = float(num.floor(timestamp))
    tt = time.gmtime(etimestamp)
    ms = (timestamp-etimestamp)*1000
    return tt,ms
        
def mystrftime(fmt=None, tt=None, milliseconds=0):
   
    if fmt is None: fmt = '%Y-%m-%d %H:%M:%S .%r'
    if tt is None: tt = time.time()
    
    fmt = fmt.replace('%r', '%03i' % int(round(milliseconds)))
    fmt = fmt.replace('%u', '%06i' % int(round(milliseconds*1000)))
    fmt = fmt.replace('%n', '%09i' % int(round(milliseconds*1000000)))
    return time.strftime(fmt, tt)
        
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
    assert len(xdata) == len(ydata)
    qpoints = QPolygonF( len(ydata) )
    vptr = qpoints.data()
    vptr.setsize(len(ydata)*8*2)
    aa = num.ndarray( shape=(len(ydata),2), dtype=num.float64, buffer=buffer(vptr))
    aa.setflags(write=True)
    aa[:,0] = xdata
    aa[:,1] = ydata
    return qpoints

class Label:
    def __init__(self, p, x, y, label_str, label_bg=None, anchor='BL', outline=False, font=None, color=None):
        text = QTextDocument()
        if font:
            text.setDefaultFont(font)
        text.setDefaultStyleSheet('span { color: %s; }' % color.name())
        text.setHtml('<span>%s</span>' % label_str)
        s = text.size()
        rect = QRectF(0., 0., s.width(), s.height())
        tx,ty =x,y
        
        if 'B' in anchor:
            ty -= rect.height()
        if 'R' in anchor:
            tx -= rect.width()
        if 'M' in anchor:
            ty -= rect.height()/2.
        if 'C' in anchor:
            tx -= rect.width()/2.
            
        rect.translate( tx, ty )
        self.rect = rect
        self.text = text
        self.outline = outline
        self.label_bg = label_bg
        self.color = color
        self.p = p

    def draw(self):
        p = self.p
        rect = self.rect
        tx = rect.left()
        ty = rect.top()

        if self.outline:
            oldpen = p.pen()
            oldbrush = p.brush()
            p.setBrush(self.label_bg)
            rect.adjust(-2.,0.,2.,0.)
            p.drawRect( rect )
            p.setPen(oldpen)
            p.setBrush(oldbrush)
            
        else:
            if self.label_bg:
                p.fillRect(rect, self.label_bg)
        
        p.translate(tx,ty)
        self.text.drawContents(p)
        p.translate(-tx,-ty)


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

    def __init__(self, low_is_none=False, high_is_none=False, low_is_zero=False, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        self.value = 0.
        self.mi = 0.
        self.ma = 1.
        self.low_is_none = low_is_none
        self.high_is_none = high_is_none
        self.low_is_zero = low_is_zero
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
            elif self.low_is_zero and float(t) == 0.0:
                value = self.mi
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

        if self.low_is_zero and self.value == self.mi:
            t = '0'

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
        
class ValControl(QObject):

    def __init__(self, low_is_none=False, high_is_none=False, low_is_zero=False, *args):
        apply(QObject.__init__, (self,) + args)
        
        self.lname = QLabel( "name" )
        self.lname.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum))
        self.lvalue = MyValueEdit( low_is_none=low_is_none, high_is_none=high_is_none, low_is_zero=low_is_zero )
        self.lvalue.setFixedWidth(100)
        self.slider = MySlider(Qt.Horizontal)
        self.slider.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.slider.setMaximum( 10000 )
        self.slider.setSingleStep( 100 )
        self.slider.setPageStep( 1000 )
        self.slider.setTickPosition( QSlider.NoTicks )
        self.slider.setFocusPolicy(Qt.ClickFocus)
        
        self.low_is_none = low_is_none
        self.high_is_none = high_is_none
        self.low_is_zero = low_is_zero

        self.connect( self.slider, SIGNAL("valueChanged(int)"),
                      self.slided )
        self.connect( self.lvalue, SIGNAL("edited(float)"),
                      self.edited )

        self.mute = False

    def widgets(self):
        return self.lname, self.lvalue, self.slider
    
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
       
    def set_range(self, mi, ma):
        if self.mi == mi and self.ma == ma:
            return

        vput = None
        if self.cursl == 0:
            vput = mi
        if self.cursl == 10000:
            vput = ma

        self.mi = mi
        self.ma = ma
        self.lvalue.setRange( self.s2v(0), self.s2v(10000) )

        if vput is not None:
            self.set_value(vput)
        else:
            if self.cur < mi:
                self.set_value(mi)
            if self.cur > ma:
                self.set_value(ma)


    def set_value(self, cur):
        if cur is None:
            if self.low_is_none:
                cur = self.mi
            elif self.high_is_none:
                cur = self.ma
    
        if cur == 0.0:
            if self.low_is_zero:
                cur = self.mi

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

        if self.cursl == 0:
            if self.low_is_none:
                cur = None

            elif self.low_is_zero:
                cur = 0.0

        if self.cursl == 10000 and self.high_is_none:
            cur = None

        self.emit(SIGNAL("valchange(PyQt_PyObject,int)"), cur, int(self.ind) )
        
class LinValControl(ValControl):
    
    def s2v(self, svalue):
        return svalue/10000. * (self.ma-self.mi) + self.mi
                
    def v2s(self, value):
        if self.ma == self.mi:
            return 0
        return int(round((value-self.mi)/(self.ma-self.mi) * 10000.))

class Progressbar:
    def __init__(self, parent, name, can_abort=True):
        self.parent = parent
        self.name = name
        self.label = QLabel(name, parent)
        self.pbar = QProgressBar(parent)
        self.aborted = False
        self.time_last_update = 0.
        if can_abort:
            self.abort_button = QPushButton('Abort', parent)
            self.parent.connect(self.abort_button, SIGNAL('clicked()'), self.abort)
        else:
            self.abort_button = False

    def widgets(self):
        widgets = [ self.label, self.bar() ]
        if self.abort_button:
            widgets.append(self.abort_button)
        return widgets
    
    def bar(self):
        return self.pbar

    def abort(self):
        self.aborted = True

class Progressbars(QFrame):
    def __init__(self, parent):
        QFrame.__init__(self, parent)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.bars = {}
        self.start_times = {}
        self.hide()

    def set_status(self, name, value):
        now = time.time()
        if name not in self.start_times:
            self.start_times[name] = now
            return False
        else:
            if now < self.start_times[name] + 1.0:
                return False

        self.start_times.get(name, 0.0)
        value = int(round(value))
        if name not in self.bars:
            if value == 100:
                return False
            self.bars[name] = Progressbar(self, name)
            self.make_layout()

        bar = self.bars[name]
        if bar.time_last_update < now - 0.1 or value == 100:
            bar.bar().setValue(value)
            bar.time_last_update = now

        if value == 100:
            del self.bars[name]
            del self.start_times[name]
            self.make_layout()
            for w in bar.widgets():
                w.setParent(None)

        return bar.aborted

    def make_layout(self):
        while True:
            c = self.layout.takeAt(0)
            if c is None:
                break

        for ibar, bar in enumerate(self.bars.values()):
            for iw, w in enumerate(bar.widgets()):
                self.layout.addWidget(w, ibar, iw)
        
        if not self.bars:
            self.hide()
        else:
            self.show()

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

    def hoover_message(self):
        return ''

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


    def draw_trace(self, viewer, p, trace, time_projection, track_projection, gain, outline_label=False):
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

    def convert_to_phase_marker(self, event=None, phasename=None, polarity=None, automatic=None, incidence_angle=None, takeoff_angle=None):

        if isinstance(self, PhaseMarker):
            return

        self.__class__ = PhaseMarker
        self._event = event
        self._phasename = phasename
        self._polarity = polarity
        self._automatic = automatic
        self._incidence_angle = incidence_angle
        self._takeoff_angle = takeoff_angle

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

    def draw_trace(self, viewer, p, trace, time_projection, track_projection, gain):
        pass
   
    def hoover_message(self):
        ev = self.get_event()
        evs = []
        for k in 'magnitude lat lon depth name region catalog'.split():
            if ev.__dict__[k] is not None and ev.__dict__[k] != '':
                if k == 'depth':
                    sv = '%g km' % (ev.depth * 0.001)
                else:
                    sv = '%s' % ev.__dict__[k]
                evs.append('%s = %s' % (k, sv))

        return ', '.join(evs) 

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

    def __init__(self, nslc_ids, tmin, tmax, kind, event=None, event_hash=None, phasename=None, polarity=None, automatic=None, incidence_angle=None, takeoff_angle=None):
        Marker.__init__(self, nslc_ids, tmin, tmax, kind)
        self._event = event
        self._event_hash = event_hash
        self._phasename = phasename
        self._polarity = polarity
        self._automatic = automatic
        self._incidence_angle = incidence_angle
        self._takeoff_angle = takeoff_angle

    def draw_trace(self, viewer, p, trace, time_projection, track_projection, gain):
        Marker.draw_trace(self, viewer, p, trace, time_projection, track_projection, gain, outline_label=(self._event is not None and self._event == viewer.get_active_event()))
         
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
        del self._event_hash
        del self._phasename
        del self._polarity
        del self._automatic
        del self._incidence_angle
        del self._takeoff_angle
        self.__class__ = Marker

    def hoover_message(self):
        toks = []
        for k in 'incidence_angle takeoff_angle polarity'.split():
            v = getattr(self, '_' + k)
            if v is not None:
                toks.append('%s = %s' % (k,v))

        return ', '.join(toks)


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
        if len(vals) == 14:
            nbasicvals = 7
        else:
            nbasicvals = 4
        nslc_ids, tmin, tmax, kind = Marker.parse_attributes(vals[1:1+nbasicvals])
       
        i = 8
        if len(vals) == 14:
            i = 11
       
        event_hash = str_to_str_or_none( vals[i-3] )
        phasename, polarity = [ str_to_str_or_none( x ) for x in vals[i:i+2] ]
        automatic = str_to_bool( vals[i+2] )
        marker = PhaseMarker( nslc_ids, tmin, tmax, kind, event=None, event_hash=event_hash,
            phasename=phasename, polarity=polarity, automatic=automatic )
        return marker

def tohex(c):
    return '%02x%02x%02x' % c

def to01(c):
    return c[0]/255., c[1]/255., c[2]/255.

class PyLab(QFrame):

    def __init__(self, parent=None):
        QFrame.__init__(self, parent)
        
        bgrgb = self.palette().color(QPalette.Window).getRgb()[:3]
        fgcolor = pyrocko.plot.tango_colors['aluminium5'] 
        dpi = 0.5*(self.logicalDpiX() + self.logicalDpiY())
    
        font = QFont()
        font.setBold(True)
        fontsize = font.pointSize()

        import matplotlib
        matplotlib.rcdefaults()
        matplotlib.rc('axes', linewidth=1)
        matplotlib.rc('xtick', direction='out')
        matplotlib.rc('ytick', direction='out')
        matplotlib.rc('xtick.major', size=8)
        matplotlib.rc('ytick.major', size=8)
        #matplotlib.rc('figure', facecolor=tohex(bgrgb), edgecolor=tohex(fgcolor))
        matplotlib.rc('figure', facecolor='white', edgecolor=tohex(fgcolor))
        matplotlib.rc('axes', facecolor='white', edgecolor=tohex(fgcolor), labelcolor=tohex(fgcolor))
        matplotlib.rc('font', family='sans-serif', weight='bold', size=fontsize)
        matplotlib.rc('text', color=tohex(fgcolor))
        matplotlib.rc('xtick', color=tohex(fgcolor))
        matplotlib.rc('ytick', color=tohex(fgcolor))
        matplotlib.rc('figure.subplot', bottom=0.15)

        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        
        layout = QGridLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        
        self.setLayout(layout)
        self.figure = Figure(dpi=100)
        self.axes_subplot = self.figure.add_subplot(111)
        ax = self.axes_subplot
        ax.set_color_cycle(map(to01, pyrocko.plot.graph_colors))

        xa = ax.get_xaxis()
        ya = ax.get_yaxis()
        for attr in ('labelpad', 'LABELPAD'):
            if hasattr(xa,attr):
                setattr(xa, attr, xa.get_label().get_fontsize())
                setattr(ya, attr, ya.get_label().get_fontsize())
                break
        canvas = FigureCanvas(self.figure)
        canvas.setParent(self)
        layout.addWidget(canvas, 0,0)

    def gca(self):
        return self.axes_subplot

    def gcf(self):
        return self.figure

