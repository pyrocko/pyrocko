#!/usr/bin/env python

'''Effective MiniSEED trace viewer.'''


# Copyright (c) 2009, Sebastian Heimann <sebastian.heimann@zmaw.de>
#
# This file is part of snuffler. For licensing information please see the file 
# COPYING which is included with snuffler.


import os
import sys
import time
import calendar
import datetime
import signal
import re
import math
import numpy as num
from itertools import izip
import scipy.stats
import tempfile
import logging
from optparse import OptionParser

import pyrocko.pile
import pyrocko.trace
import pyrocko.util
import pyrocko.plot

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import *
from PyQt4.QtSvg import *
#QWidget = QGLWidget

logger = logging.getLogger('pyrocko.pile_viewer')

class Global:
    sacflag = False
    appOnDemand = None

gap_lap_tolerance = 5.

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

class ObjectStyle(object):
    def __init__(self, frame_pen, fill_brush):
        self.frame_pen = frame_pen
        self.fill_brush = fill_brush

box_styles = []
box_alpha = 100
for color in 'orange skyblue butter chameleon chocolate plum scarletred'.split():
    box_styles.append(ObjectStyle(
        QPen(QColor(*pyrocko.plot.tango_colors[color+'3'])),
        QBrush(QColor(*(pyrocko.plot.tango_colors[color+'1']+(box_alpha,)))),        
    ))
    
sday   = 60*60*24     # \ 
smonth = 60*60*24*30  #   > only used as approx. intervals...
syear  = 60*60*24*365 # /

acceptable_tincs = num.array([1, 2, 5, 10, 20, 30, 60, 60*5, 60*10, 60*20, 60*30, 60*60, 
                     60*60*3, 60*60*6, 60*60*12, sday, smonth, syear ],dtype=num.float)



def neic_earthquakes(tmin, tmax, magnitude_range=(5,9.9)):
    
    import urllib2
    
    stt = time.gmtime(tmin)
    ett = time.gmtime(tmax)
    syear, smonth, sday = stt[:3]
    eyear, emonth, eday = ett[:3]
    
    l = 'http://neic.usgs.gov/cgi-bin/epic/epic.cgi?SEARCHMETHOD=1&FILEFORMAT=4&SEARCHRANGE=HH&SYEAR=%i&SMONTH=%i&SDAY=%i&EYEAR=%i&EMONTH=%i&EDAY=%i&LMAG=%f&UMAG=%f&NDEP1=&NDEP2=&IO1=&IO2=&SLAT2=0.0&SLAT1=0.0&SLON2=0.0&SLON1=0.0&CLAT=0.0&CLON=0.0&CRAD=0&SUBMIT=Submit+Search' % (syear, smonth, sday, eyear, emonth, eday, magnitude_range[0], magnitude_range[1])

    response = urllib2.urlopen(l)
    
    html = response.read()
    
    markers = []
    for line in html.splitlines():
        toks = line.split()
        if not len(toks) == 12: continue
        if not toks[0].startswith('PDE'): continue
        datestr = ' '.join(toks[1:4]+[toks[4][:6]])
        tt = time.strptime(datestr, '%Y %m %d %H%M%S')
        
        t = calendar.timegm(tt)
        markers.append(EventMarker(t, float(toks[8])))
    return markers

def get_working_system_time_range():
    now = time.time()
    hi = now
    for ignore in range(200):
        now += syear
        try:
            tt = time.gmtime(now)
            time.strftime('', tt)
            hi = now
        except:
            break
        
    now = time.time()
    lo = now
    for ignore in range(200):    
        now -= syear
        try:
            tt = time.gmtime(now)
            time.strftime('',tt)
            lo = now
        except:
            break
    return lo, hi

working_system_time_range = get_working_system_time_range()

def is_working_time(t):
    return working_system_time_range[0] <= t and  t <= working_system_time_range[1]
        

def fancy_time_ax_format(inc):
    l0_fmt_brief = ''
    l2_fmt = ''
    l2_trig = 0
    if inc < 1:
        l0_fmt = '.%r'
        l0_center = False
        l1_fmt = '%H:%M:%S'
        l1_trig = 6
        l2_fmt = '%b %d, %Y'
        l2_trig = 3
    elif inc < 60:
        l0_fmt = '%H:%M:%S'
        l0_center = False
        l1_fmt = '%b %d, %Y'
        l1_trig = 3
    elif inc < 3600:
        l0_fmt = '%H:%M'
        l0_center = False
        l1_fmt = '%b %d, %Y'
        l1_trig = 3
    elif inc < sday:
        l0_fmt = '%H:%M'
        l0_center = False
        l1_fmt = '%b %d, %Y'
        l1_trig = 3
    elif inc < smonth:
        l0_fmt = '%a %d'
        l0_fmt_brief = '%d'
        l0_center = True
        l1_fmt = '%b, %Y'
        l1_trig = 2
    elif inc < syear:
        l0_fmt = '%b'
        l0_center = True
        l1_fmt = '%Y'
        l1_trig = 1
    else:
        l0_fmt = '%Y'
        l0_center = False
        l1_fmt = ''
        l1_trig = 0
        
    return l0_fmt, l0_fmt_brief, l0_center, l1_fmt, l1_trig, l2_fmt, l2_trig

def day_start(timestamp):
   tt = time.gmtime(timestamp)
   tts = tt[0:3] + (0,0,0) + tt[6:9]
   return calendar.timegm(tts)

def month_start(timestamp):
   tt = time.gmtime(timestamp)
   tts = tt[0:2] + (1,0,0,0) + tt[6:9]
   return calendar.timegm(tts)

def year_start(timestamp):
    tt = time.gmtime(timestamp)
    tts = tt[0:1] + (1,1,0,0,0) + tt[6:9]
    return calendar.timegm(tts)

def gmtime_x(timestamp):
    etimestamp = math.floor(timestamp)
    tt = time.gmtime(etimestamp)
    ms = (timestamp-etimestamp)*1000
    return tt,ms
        
def time_nice_value(inc0):
    if inc0 < acceptable_tincs[0]:
        return pyrocko.plot.nice_value(inc0)
    elif inc0 > acceptable_tincs[-1]:
        return pyrocko.plot.nice_value(inc0/syear)*syear
    else:
        i = num.argmin(num.abs(acceptable_tincs-inc0))
        return acceptable_tincs[i]

class TimeScaler(pyrocko.plot.AutoScaler):
    def __init__(self):
        pyrocko.plot.AutoScaler.__init__(self)
        self.mode = 'min-max'
    
    def make_scale(self, data_range):
        assert self.mode in ('min-max', 'off'), 'mode must be "min-max" or "off" for TimeScaler'
        
        data_min = min(data_range)
        data_max = max(data_range)
        is_reverse = (data_range[0] > data_range[1])
        
        mi, ma = data_min, data_max
        nmi = mi
        if self.mode != 'off':
            nmi = mi - self.space*(ma-mi)
            
        nma = ma
        if self.mode != 'off':
            nma = ma + self.space*(ma-mi)
             
        mi, ma = nmi, nma
        
        if mi == ma and a != 'off':
            mi -= 1.0
            ma += 1.0
        
        mi = max(working_system_time_range[0],mi)
        ma = min(working_system_time_range[1],ma)
        
        # make nice tick increment
        if self.inc is not None:
            inc = self.inc
        else:
            if self.approx_ticks > 0.:
                inc = time_nice_value( (ma-mi)/self.approx_ticks )
            else:
                inc = time_nice_value( (ma-mi)*10. )
        
        if inc == 0.0:
            inc = 1.0
        
        if is_reverse:
            return ma, mi, -inc
        else:
            return mi, ma, inc


    def make_ticks(self, data_range):
        mi, ma, inc = self.make_scale(data_range)
        
        is_reverse = False
        if inc < 0:
            mi, ma, inc = ma, mi, -inc
            is_reverse = True
            
        ticks = []
        
        if inc < sday:
            mi_day = day_start(max(mi, working_system_time_range[0]+sday*1.5))
            base = mi_day+math.ceil((mi-mi_day)/inc)*inc
            base_day = mi_day
            i = 0
            while True:
                tick = base+i*inc
                if tick > ma: break
                tick_day = day_start(tick)
                if tick_day > base_day:
                    base_day = tick_day
                    base = base_day
                    i = 0
                else:
                    ticks.append(tick)
                    i += 1
                    
        elif inc < smonth:
            mi_day = day_start(max(mi, working_system_time_range[0]+sday*1.5))
            dt_base = datetime.datetime(*time.gmtime(mi_day)[:6])
            delta = datetime.timedelta(days=int(round(inc/sday)))
            if mi_day == mi: dt_base += delta
            i = 0
            while True:
                current = dt_base + i*delta
                tick = calendar.timegm(current.timetuple())
                if tick > ma: break
                ticks.append(tick)
                i += 1
            
        elif inc < syear:
            mi_month = month_start(max(mi, working_system_time_range[0]+smonth*1.5))
            y,m = time.gmtime(mi_month)[:2]
            while True:
                tick = calendar.timegm((y,m,1,0,0,0))
                m += 1
                if m > 12: y,m = y+1,1
                if tick > ma: break
                if tick >= mi: ticks.append(tick)
        
        else:
            mi_year = year_start(max(mi, working_system_time_range[0]+syear*1.5))
            incy = int(round(inc/syear))
            y = int(math.floor(time.gmtime(mi_year)[0]/incy)*incy)
            
            while True:
                tick = calendar.timegm((y,1,1,0,0,0))
                y += incy
                if tick > ma: break
                if tick >= mi: ticks.append(tick)
        
        if is_reverse: ticks.reverse()
        return ticks, inc

def need_l1_tick(tt, ms, l1_trig):
    return (0,1,1,0,0,0)[l1_trig:] == tt[l1_trig:6] and ms == 0.0
    
 
def mystrftime(fmt=None, tt=None, milliseconds=0):
   
    if fmt is None: fmt = '%Y-%m-%d %H:%M:%S .%r'
    if tt is None: tt = time.time()
    
    fmt2 = fmt.replace('%r', '%03i' % int(round(milliseconds)))
    return time.strftime(fmt2, tt)
        
        
def myctime(timestamp):
    tt, ms = gmtime_x(timestamp)
    return mystrftime(None, tt, ms)
    

def tick_to_labels(tick, inc):
    tt, ms = gmtime_x(tick)
    l0_fmt, l0_fmt_brief, l0_center, l1_fmt, l1_trig, l2_fmt, l2_trig = fancy_time_ax_format(inc)
    l0 = mystrftime(l0_fmt, tt, ms)
    l0_brief = mystrftime(l0_fmt_brief, tt, ms)
    l1, l2 = None, None
    if need_l1_tick(tt, ms, l1_trig):
        l1 = mystrftime(l1_fmt, tt, ms)
    if need_l1_tick(tt, ms, l2_trig):
        l2 = mystrftime(l2_fmt, tt, ms)
        
    return l0, l0_brief, l0_center, l1, l2
    
def l1_l2_tick(tick, inc):
    tt, ms = gmtime_x(tick)
    l0_fmt, l0_fmt_brief, l0_center, l1_fmt, l1_trig, l2_fmt, l2_trig = fancy_time_ax_format(inc)
    l1 = mystrftime(l1_fmt, tt, ms)
    l2 = mystrftime(l2_fmt, tt, ms)
    return l1, l2
    
class TimeAx(TimeScaler):
    def __init__(self, *args):
        TimeScaler.__init__(self, *args)
    
    
    def drawit( self, p, xprojection, yprojection ):
        pen = QPen(QColor(*pyrocko.plot.tango_colors['aluminium5']),1)
        p.setPen(pen)
        font = QFont()
        font.setBold(True)
        p.setFont(font)
        fm = p.fontMetrics()
        ticklen = 10
        pad = 10
        tmin, tmax = xprojection.get_in_range()
        ticks, inc = self.make_ticks((tmin,tmax))
        l1_hits = 0
        l2_hits = 0
        
        vmin, vmax = yprojection(0), yprojection(ticklen)
        uumin, uumax = xprojection.get_out_range()
        first_tick_with_label = None
        for tick in ticks:
            umin = xprojection(tick)
            
            umin_approx_next = xprojection(tick+inc)
            umax = xprojection(tick)
            
            pinc_approx = umin_approx_next - umin
            
            p.drawLine(QPointF(umin, vmin), QPointF(umax, vmax))
            l0, l0_brief, l0_center, l1, l2 = tick_to_labels(tick, inc)
            
            if l0_center:
                ushift = (umin_approx_next-umin)/2.
            else:
                ushift = 0.
            
            for l0x in (l0, l0_brief, ''):
                label0 = QString(l0x)
                rect0 = fm.boundingRect( label0 )
                if rect0.width() <= pinc_approx*0.9: break
            
            if uumin+pad < umin-rect0.width()/2.+ushift and umin+rect0.width()/2.+ushift < uumax-pad:
                if first_tick_with_label is None:
                    first_tick_with_label = tick
                p.drawText( QPointF(umin-rect0.width()/2.+ushift, vmin+rect0.height()+ticklen), label0 )
            
            if l1:
                label1 = QString(l1)
                rect1 = fm.boundingRect( label1 )
                if uumin+pad < umin-rect1.width()/2. and umin+rect1.width()/2. < uumax-pad:
                    p.drawText( QPointF(umin-rect1.width()/2., vmin+rect0.height()+rect1.height()+ticklen), label1 )
                    l1_hits += 1
                
            if l2:
                label2 = QString(l2)
                rect2 = fm.boundingRect( label2 )
                if uumin+pad < umin-rect2.width()/2. and umin+rect2.width()/2. < uumax-pad:
                    p.drawText( QPointF(umin-rect2.width()/2., vmin+rect0.height()+rect1.height()+rect2.height()+ticklen), label2 )
                    l2_hits += 1
        
        if first_tick_with_label is None:
            first_tick_with_label = tmin
            
        l1, l2 = l1_l2_tick(first_tick_with_label, inc)
        
        if l1_hits == 0 and l1:
            label1 = QString(l1)
            rect1 = fm.boundingRect( label1 )
            p.drawText( QPointF(uumin+pad, vmin+rect0.height()+rect1.height()+ticklen), label1 )
            l1_hits += 1
        
        if l2_hits == 0 and l2:
            label2 = QString(l2)
            rect2 = fm.boundingRect( label2 )
            p.drawText( QPointF(uumin+pad, vmin+rect0.height()+rect1.height()+rect2.height()+ticklen), label2 )
                
        v = yprojection(0)
        p.drawLine(QPointF(uumin, v), QPointF(uumax, v))
        
class Projection(object):
    def __init__(self):
        self.xr = 0.,1.
        self.ur = 0.,1.
        
    def set_in_range(self, xmin, xmax):
        if xmax == xmin: xmax = xmin + 1.
        self.xr = float(xmin), float(xmax)

    def get_in_range(self):
        return self.xr

    def set_out_range(self, umin, umax):
        if umax == umin: umax = umin + 1.
        self.ur = umin, umax
        
    def get_out_range(self):
        return self.ur
        
    def __call__(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return umin + (x-xmin)*((umax-umin)/(xmax-xmin))
        
    def rev(self, u):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return xmin + (u-umin)*((xmax-xmin)/(umax-umin))


# XXX currently unused:
class TraceOverview(object):
    def __init__(self, mstrace, file_abspath, style):
        self.mstrace = mstrace
        self.file_abspath = file_abspath
        self.style = style
        
    def drawit(self, p, time_projection, v_projection):
        
        if self.mstrace.overlaps(*time_projection.get_in_range()):

            font = QFont()
            font.setBold(True)
            font.setPointSize(6)
            p.setFont(font)
            fm = p.fontMetrics()
            dtmin = time_projection(self.mstrace.tmin)
            dtmax = time_projection(self.mstrace.tmax)
            
            dvmin = v_projection(0.)
            dvmax = v_projection(1.)
            
            rect = QRectF( dtmin, dvmin, dtmax-dtmin, dvmax-dvmin )
            p.fillRect(rect, self.style.fill_brush)
            p.setPen(self.style.frame_pen)
            p.drawRect(rect)
            
            fn_label = QString(self.file_abspath)
            label_rect = fm.boundingRect( fn_label )
            
            if label_rect.width() < 0.9*rect.width():
                p.drawText( QPointF(rect.left()+5, rect.bottom()-5), fn_label )
    
    def get_mstrace(self):
        return self.mstrace

def add_radiobuttongroup(menu, menudef, obj, target):
    group = QActionGroup(menu)
    menuitems = []
    for l, v in menudef:
        k = QAction(l, menu)
        group.addAction(k)
        menu.addAction(k)
        k.setCheckable(True)
        obj.connect(group, SIGNAL('triggered(QAction*)'), target)
        menuitems.append((k,v))
            
    menuitems[0][0].setChecked(True)
    return menuitems

class MarkerParseError(Exception):
    pass

class Marker:
    
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
                
            elif len(toks) in (7,8):
                tmin = parsedate(*toks[:3])
                tmax = parsedate(*toks[3:6])
            
            else:
                fail()
            
            if len(toks) in (5,8):
                kind = int(toks[-2])
            else:
                kind = int(toks[-1])
                
            nslc_ids = []
            if len(toks) in (5,8):
                for snslc in toks[-1].split(','):
                    nslc = snslc.split('.')
                    if len(nslc) != 4:
                        fail()
                        
                    nslc_ids.append(tuple(nslc))
                
        except MarkerParseError:
            fail()
    
        return Marker(nslc_ids, tmin, tmax, kind=kind)
    
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
        
    def __str__(self):
        traces = ', '.join( [ '.'.join(nslc_id) for nslc_id in self.nslc_ids ] )
        st = myctime
        if self.tmin == self.tmax:
            return '%s %i %s' % (st(self.tmin), self.kind, traces)
        else:
            return '%s %s %g %i %s' % (st(self.tmin), st(self.tmax), self.tmax-self.tmin, self.kind, traces)
        
    def select_color(self, colorlist):
        cl = lambda x: colorlist[(self.kind*3+x)%len(colorlist)]
        if self.selected:
            return cl(0)
        if self.alerted:
            return cl(1)
        return cl(2)
            
    def draw(self, p, time_projection, y_projection):
        color = self.select_color(self.color_b)            
        pen = QPen(QColor(*color))
        if self.selected or self.alerted:
            pen.setStyle(Qt.CustomDashLine)
            pat = [5.,3.]
            pen.setDashPattern(pat)
        pen.setWidth(2)
        p.setPen(pen)
        
        def drawline(t):
            u = time_projection(t)
            v0, v1 = y_projection.get_out_range()
            line = QLine(u,v0,u,v1)
            p.drawLine(line)
            
        if self.selected or self.alerted or not self.nslc_ids:
            drawline(self.tmin)
            drawline(self.tmax)

    def draw_trace(self, p, trace, time_projection, track_projection, gain):
        if self.nslc_ids and trace.nslc_id not in self.nslc_ids: return
        
        color = self.select_color(self.color_b)
        pen = QPen(QColor(*color))
        pen.setWidth(2)
        p.setPen(pen)
        
        def drawpoint(t,y):
            u = time_projection(t)
            v = track_projection(y)
            rect = QRect(u-2,v-2,4,4)
            p.drawRect(rect)
            
        def drawline(t):
            u = time_projection(t)
            v0, v1 = track_projection.get_out_range()
            line = QLine(u,v0,u,v1)
            p.drawLine(line)

        try:
            snippet = trace.chop(self.tmin, self.tmax, inplace=False, include_last=True, snap=(math.ceil,math.floor))
            udata = time_projection(snippet.get_xdata())
            vdata = track_projection( gain*snippet.get_ydata() )
            qpoints = make_QPolygonF( udata, vdata )
            p.drawPolyline( qpoints )
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
        try: drawpoint(self.tmin, trace.interpolate(self.tmin))
        except IndexError: pass
        try: drawpoint(self.tmax, trace.interpolate(self.tmax))
        except IndexError: pass            

class EventMarker(Marker):
    def __init__(self, time, magnitude):
        Marker.__init__(self, '', time, time)
        

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
        

def MakePileOverviewClass(base):
    
    class PileOverview(base):
        def __init__(self, pile, ntracks_shown_max, *args):
            apply(base.__init__, (self,) + args)
            
            self.pile = pile
            self.ax_height = 80
            
            self.click_tolerance = 5
            
            self.ntracks_shown_max = ntracks_shown_max
            self.ntracks = 0
            self.shown_tracks_range = None
            self.track_start = None
            self.track_trange = None
            
            self.lowpass = None
            self.highpass = None
            self.gain = 1.0
            self.rotate = 0.0
            self.markers = []
            self.picking_down = None
            self.picking = None
            self.floating_marker = None
            self.markers = []
            self.ignore_releases = 0
            self.message = None
            self.reloaded = False
            
            self.tax = TimeAx()
            self.setBackgroundRole( QPalette.Base )
            self.setAutoFillBackground( True )
            poli = QSizePolicy( QSizePolicy.Expanding, QSizePolicy.Expanding )
            self.setSizePolicy( poli )
            self.setMinimumSize(300,200)
            self.setFocusPolicy( Qt.StrongFocus )
    
            self.menu = QMenu(self)
             
            self.menuitem_pick = QAction('Pick', self.menu)
            self.menu.addAction(self.menuitem_pick)
            self.connect( self.menuitem_pick, SIGNAL("triggered(bool)"), self.start_picking )
            
            self.menuitem_pick = QAction('Write picks', self.menu)
            self.menu.addAction(self.menuitem_pick)
            self.connect( self.menuitem_pick, SIGNAL("triggered(bool)"), self.write_picks )
            
            self.menuitem_pick = QAction('Read picks', self.menu)
            self.menu.addAction(self.menuitem_pick)
            self.connect( self.menuitem_pick, SIGNAL("triggered(bool)"), self.read_picks )
            
            self.menu.addSeparator()
            
            self.menuitem_neic = QAction('NEIC catalog events 5+', self.menu)
            self.menu.addAction(self.menuitem_neic)
            self.connect( self.menuitem_neic, SIGNAL("triggered(bool)"), self.get_neic_events )
            
            self.menu.addSeparator()
    
            menudef = [
                ('Indivdual Scale',            lambda tr: (tr.network, tr.station, tr.location, tr.channel)),
                ('Common Scale',               lambda tr: None),
                ('Common Scale per Station',   lambda tr: (tr.network, tr.station)),
                ('Common Scale per Component', lambda tr: (tr.channel)),
            ]
            
            self.menuitems_scaling = add_radiobuttongroup(self.menu, menudef, self, self.scalingmode_change)
            self.scaling_key = self.menuitems_scaling[0][1]
            
            self.menu.addSeparator()
            
            menudef = [
                ('Scaling based on Minimum and Maximum', 'minmax'),
                ('Scaling based on Mean +- 2 x Std. Deviation', 2),
                ('Scaling based on Mean +- 4 x Std. Deviation', 4),
            ]
            
            self.menuitems_scalingbase = add_radiobuttongroup(self.menu, menudef, self, self.scalingbase_change)
            self.scalingbase = self.menuitems_scalingbase[0][1]
            
            self.menu.addSeparator()
            
            menudef = [
                ('Sort by Network, Station, Location, Channel', 
                    ( lambda tr: tr.nslc_id,     # gathering
                    lambda a,b: cmp(a,b),      # sorting
                    lambda tr: tr.location )),  # coloring
                ('Sort by Network, Station, Channel, Location', 
                    ( lambda tr: tr.nslc_id, 
                    lambda a,b: cmp((a[0],a[1],a[3],a[2]), (b[0],b[1],b[3],b[2])),
                    lambda tr: tr.channel )),
                ('Sort by Station, Network, Channel, Location', 
                    ( lambda tr: tr.nslc_id, 
                    lambda a,b: cmp((a[1],a[0],a[3],a[2]), (b[1],b[0],b[3],b[2])),
                    lambda tr: tr.channel )),
                ('Sort by Location, Network, Station, Channel', 
                    ( lambda tr: tr.nslc_id, 
                    lambda a,b: cmp((a[2],a[0],a[1],a[3]), (b[2],b[0],b[1],b[3])),
                    lambda tr: tr.channel )),
                ('Sort by Network, Station, Channel',
                    ( lambda tr: (tr.network, tr.station, tr.channel),
                    lambda a,b: cmp(a,b),
                    lambda tr: tr.location )),
                ('Sort by Station, Network, Channel',
                    ( lambda tr: (tr.station, tr.network, tr.channel),
                    lambda a,b: cmp(a,b),
                    lambda tr: tr.location )),
            ]
            self.menuitems_sorting = add_radiobuttongroup(self.menu, menudef, self, self.sortingmode_change)
            
            self.menu.addSeparator()
            
            self.menuitem_antialias = QAction('Antialiasing', self.menu)
            self.menuitem_antialias.setCheckable(True)
            self.menu.addAction(self.menuitem_antialias)
            
            self.menuitem_cliptraces = QAction('Clip Traces', self.menu)
            self.menuitem_cliptraces.setCheckable(True)
            self.menuitem_cliptraces.setChecked(True)
            self.menu.addAction(self.menuitem_cliptraces)
            
            self.menuitem_showboxes = QAction('Show Boxes', self.menu)
            self.menuitem_showboxes.setCheckable(True)
            self.menuitem_showboxes.setChecked(True)
            self.menu.addAction(self.menuitem_showboxes)
            
            self.menuitem_colortraces = QAction('Color Traces', self.menu)
            self.menuitem_colortraces.setCheckable(True)
            self.menuitem_colortraces.setChecked(False)
            self.menu.addAction(self.menuitem_colortraces)
            
            self.menuitem_showscalerange = QAction('Show Scale Range', self.menu)
            self.menuitem_showscalerange.setCheckable(True)
            self.menu.addAction(self.menuitem_showscalerange)
    
            self.menuitem_allowdownsampling = QAction('Allow Downsampling', self.menu)
            self.menuitem_allowdownsampling.setCheckable(True)
            self.menuitem_allowdownsampling.setChecked(True)
            self.menu.addAction(self.menuitem_allowdownsampling)
            
            self.menuitem_degap = QAction('Allow Degapping', self.menu)
            self.menuitem_degap.setCheckable(True)
            self.menuitem_degap.setChecked(True)
            self.menu.addAction(self.menuitem_degap)
            
            self.menuitem_fft_filtering = QAction('FFT Filtering', self.menu)
            self.menuitem_fft_filtering.setCheckable(True)
            self.menuitem_fft_filtering.setChecked(False)
            self.menu.addAction(self.menuitem_fft_filtering)
            
            self.menuitem_lphp = QAction('Bandpass is Lowpass + Highpass', self.menu)
            self.menuitem_lphp.setCheckable(True)
            self.menuitem_lphp.setChecked(True)
            self.menu.addAction(self.menuitem_lphp)
            
            self.menuitem_watch = QAction('Watch Files', self.menu)
            self.menuitem_watch.setCheckable(True)
            self.menuitem_watch.setChecked(False)
            self.menu.addAction(self.menuitem_watch)
            
            self.menu.addSeparator()
            
            self.snufflings_menu = QMenu('Snufflings', self.menu)
            self.menu.addMenu(self.snufflings_menu)
            
            self.menuitem_reload = QAction('Reload snufflings', self.menu)
            self.menu.addAction(self.menuitem_reload)
            self.connect( self.menuitem_reload, SIGNAL("triggered(bool)"), self.setup_snufflings )
            self.menu.addSeparator()

            self.menuitem_print = QAction('Print', self.menu)
            self.menu.addAction(self.menuitem_print)
            self.connect( self.menuitem_print, SIGNAL("triggered(bool)"), self.printit )
            
            self.menuitem_svg = QAction('Save as SVG', self.menu)
            self.menu.addAction(self.menuitem_svg)
            self.connect( self.menuitem_svg, SIGNAL("triggered(bool)"), self.savesvg )
            
            self.menuitem_close = QAction('Close', self.menu)
            self.menu.addAction(self.menuitem_close)
            self.connect( self.menuitem_close, SIGNAL("triggered(bool)"), self.myclose )
            
            self.menu.addSeparator()
    
            self.menuitem_fuckup = QAction("Snuffler sucks! It can't do this and that...", self.menu)
            self.menu.addAction(self.menuitem_fuckup)
            self.connect( self.menuitem_fuckup, SIGNAL("triggered(bool)"), self.fuck )
    
            deltats = self.pile.get_deltats()
            if deltats:
                self.min_deltat = min(deltats)
            else:
                self.min_deltat = 0.01
                
            self.time_projection = Projection()
            self.set_time_range(self.pile.tmin, self.pile.tmax)
            self.time_projection.set_out_range(0., self.width())
                
            self.set_gathering()
    
            self.track_to_screen = Projection()
            self.track_to_nslc_ids = {}
        
            self.old_vec = None
            self.old_processed_traces = None
            
            self.timer = QTimer( self )
            self.connect( self.timer, SIGNAL("timeout()"), self.periodical ) 
            self.timer.setInterval(1000)
            self.timer.start()
            self.pile.add_listener(self)
            self.trace_styles = {}
            self.determine_box_styles()
            self.setMouseTracking(True)
            
            user_home_dir = os.environ['HOME']

            self.snuffling_paths = [ os.path.join(user_home_dir, '.snufflings') ] 
            self.setup_snufflings()
            
            
        def get_snufflings(self):
            snufflings = []
            for path in self.snuffling_paths:
                if os.path.isdir(path):
                    for fn in os.listdir(path):
                        if fn.endswith('.py'):
                            mod = SnufflingModule(path, fn[:-3])
                            snufflings.extend(mod.snufflings())
                            
            return snufflings
            
        def setup_snufflings(self):
            self.snuffling_hooks = []
            self.snufflings_menu.clear()
            for snuffling in self.get_snufflings():
                item = QAction(snuffling.get_name(), self.menu)
                self.snufflings_menu.addAction(item)
                def hook():
                    self.call_snuffling(snuffling)
                    
                self.snuffling_hooks.append(hook)
                self.connect( item, SIGNAL("triggered(bool)"), hook )
        
                 
        def call_snuffling(self, snuffling):
            snuffling.call(self)
        
        def periodical(self):
            if self.menuitem_watch.isChecked():
                if self.pile.reload_modified():
                    self.update()
    
        def get_pile(self):
            return self.pile
        
        def pile_changed(self, what):
            self.sortingmode_change()
            self.determine_box_styles()
            
        def get_neic_events(self):
            self.set_markers(neic_earthquakes(self.pile.tmin, self.pile.tmax, magnitude_range=(5.,9.9)))
    
        def set_gathering(self, gather=None, order=None, color=None):
            if gather is None:
                gather = lambda tr: tr.nslc_id
                
            if order is None:
                order = lambda a,b: cmp(a, b)
            
            if color is None:
                color = lambda tr: tr.location
            
            self.gather = gather
            keys = self.pile.gather_keys(gather)
            self.color_gather = color
            self.color_keys = self.pile.gather_keys(color)
            previous_ntracks = self.ntracks
            self.ntracks = len(keys)
            if self.shown_tracks_range is None or previous_ntracks < self.ntracks:
                self.shown_tracks_range = 0, self.ntracks
                
            l, h = self.shown_tracks_range
            if l >= self.ntracks: l = self.ntracks-1
            if h > self.ntracks: h = self.ntracks
            
            self.shown_tracks_range = l, h
            self.shown_tracks_start = float(self.shown_tracks_range[0])
    
            self.track_keys = sorted(keys, cmp=order)
            self.key_to_row = dict([ (key, i) for (i,key) in enumerate(self.track_keys) ])
            
            inrange = lambda x,r: r[0] <= x and x < r[1]
            self.trace_selector = lambda trace: inrange(self.key_to_row[self.gather(trace)], self.shown_tracks_range)
        
            if self.tmin == working_system_time_range[0] and self.tmax == working_system_time_range[1]:
                self.set_time_range(self.pile.tmin, self.pile.tmax)
        
        def set_time_range(self, tmin, tmax):
            self.tmin, self.tmax = tmin, tmax
            
            if self.tmin > self.tmax:
                self.tmin, self.tmax = self.tmax, self.tmin
                
            if self.tmin == self.tmax:
                self.tmin -= 1.
                self.tmax += 1.
            
            self.tmin = max(working_system_time_range[0], self.tmin)
            self.tmax = min(working_system_time_range[1], self.tmax)
                    
            if (self.tmax - self.tmin < self.min_deltat):
                m = (self.tmin + self.tmax) / 2.
                self.tmin = m - self.min_deltat/2.
                self.tmax = m + self.min_deltat/2.
                
            self.time_projection.set_in_range(tmin,tmax)
        
        def get_time_range(self):
            return self.tmin, self.tmax
        
        def ypart(self, y):
            if y < self.ax_height:
                return -1
            elif y > self.height()-self.ax_height:
                return 1
            else:
                return 0
    
        def write_picks(self):
            fn = QFileDialog.getSaveFileName(self,)
            f = open(fn,'w')
            for marker in self.markers:
                f.write("%s\n" % marker)
            f.close()
                
            
        def read_picks(self):
            fn = QFileDialog.getOpenFileName(self,)
            f = open(fn, 'r')
            for iline, line in enumerate(f):
                sline = line.strip()
                if not sline or sline.startswith('#'):
                    continue
                try:
                    m = Marker.from_string(sline)
                    self.markers.append(m)
                    
                except MarkerParseError:
                    logger.warn('Invalid marker definition in line %i of file "%s"' % (iline+1, fn))
                    
            f.close()
            
        def add_marker(self, marker):
            self.markers.append(marker)
                
        def set_markers(self, markers):
            self.markers = markers
    
        def mousePressEvent( self, mouse_ev ):
            #self.setMouseTracking(False)
            point = self.mapFromGlobal(mouse_ev.globalPos())

            if mouse_ev.button() == Qt.LeftButton:
                marker = self.marker_under_cursor(point.x(), point.y())
                if self.picking:
                    if self.picking_down is None:
                        self.picking_down = self.time_projection.rev(mouse_ev.x()), mouse_ev.y()
                elif marker is not None:
                    if not (mouse_ev.modifiers() & Qt.ShiftModifier):
                        self.deselect_all()
                    marker.set_selected(True)
                    self.update()
                else:
                    self.track_start = mouse_ev.x(), mouse_ev.y()
                    self.track_trange = self.tmin, self.tmax
            
            if mouse_ev.button() == Qt.RightButton:
                self.menu.exec_(QCursor.pos())
            self.update_status()
    
        def mouseReleaseEvent( self, mouse_ev ):
            if self.ignore_releases:
                self.ignore_releases -= 1
                return
            
            if self.picking:
                self.stop_picking(mouse_ev.x(), mouse_ev.y())
            self.track_start = None
            self.track_trange = None
            self.update_status()
            
        def mouseDoubleClickEvent(self, mouse_ev):
            self.start_picking(None)
            self.ignore_releases = 1
    
        def mouseMoveEvent( self, mouse_ev ):
            point = self.mapFromGlobal(mouse_ev.globalPos())
    
            if self.picking:
                self.update_picking(point.x(),point.y())
           
            elif self.track_start is not None:
                x0, y0 = self.track_start
                dx = (point.x() - x0)/float(self.width())
                dy = (point.y() - y0)/float(self.height())
                if self.ypart(y0) == 1: dy = 0
                
                tmin0, tmax0 = self.track_trange
                
                scale = math.exp(-dy*5.)
                dtr = scale*(tmax0-tmin0) - (tmax0-tmin0)
                frac = x0/float(self.width())
                dt = dx*(tmax0-tmin0)*scale
                
                self.set_time_range(tmin0 - dt - dtr*frac, tmax0 - dt + dtr*(1.-frac))
    
                self.update()
            else:
                self.hoovering(point.x(),point.y())
                
            self.update_status()
       
       
        def nslc_ids_under_cursor(self, x,y):
            ftrack = self.track_to_screen.rev(y)
            nslc_ids = self.get_nslc_ids_for_track(ftrack)
            return nslc_ids
       
        def marker_under_cursor(self, x,y):
            mouset = self.time_projection.rev(x)
            deltat = (self.tmax-self.tmin)*self.click_tolerance/self.width()
            relevant_nslc_ids = None
            for marker in self.markers:
                if (abs(mouset-marker.get_tmin()) < deltat or 
                    abs(mouset-marker.get_tmax()) < deltat):
                    
                    if relevant_nslc_ids is None:
                        relevant_nslc_ids = self.nslc_ids_under_cursor(x,y)
                        
                    for nslc_id in marker.get_nslc_ids():
                        if nslc_id in relevant_nslc_ids:
                            return marker
       
        def hoovering(self, x,y):
            mouset = self.time_projection.rev(x)
            deltat = (self.tmax-self.tmin)*self.click_tolerance/self.width()
            needupdate = False
            haveone = False
            relevant_nslc_ids = self.nslc_ids_under_cursor(x,y)
            
            for marker in self.markers:
                state = abs(mouset-marker.get_tmin()) < deltat or \
                        abs(mouset-marker.get_tmax()) < deltat and not haveone
                
                if state:
                    xstate = False
                    for nslc_id in marker.get_nslc_ids():
                        if nslc_id in relevant_nslc_ids:
                            xstate = True
                            
                    state = xstate
                    
                if state:
                    haveone = True
                oldstate = marker.is_alerted()
                if oldstate != state:
                    needupdate = True
                    marker.set_alerted(state)
                    
            if needupdate:
                self.update()
                
        def keyPressEvent(self, key_event):
            dt = self.tmax - self.tmin
            tmid = (self.tmin + self.tmax) / 2.
            
            if key_event.text() == ' ':
                self.set_time_range(self.tmin+dt, self.tmax+dt)
            
            elif key_event.text() == 'b':
                dt = self.tmax - self.tmin
                self.set_time_range(self.tmin-dt, self.tmax-dt)
            
            elif key_event.text() == 'n':
                for marker in sorted(self.markers, cmp=lambda a,b: cmp(a.tmin,b.tmin)):
                    t = marker.tmin
                    if t > tmid:
                        self.deselect_all()
                        marker.set_selected(True)
                        self.set_time_range(t-dt/2.,t+dt/2.)
                        break
                
            elif key_event.text() == 'p':
                for marker in sorted(self.markers, cmp=lambda a,b: cmp(b.tmin,a.tmin)):
                    t = marker.tmin
                    if t < tmid:
                        self.deselect_all()
                        marker.set_selected(True)
                        self.set_time_range(t-dt/2.,t+dt/2.)
                        break
                        
            elif key_event.text() == 'q':
                self.myclose()
    
            elif key_event.text() == 'r':
                if self.pile.reload_modified():
                    self.reloaded = True
    
            elif key_event.key() == Qt.Key_Backspace:
                markers = []
                for marker in self.markers:
                    if not marker.is_selected():
                        markers.append(marker)
                        
                self.markers = markers
    
            elif key_event.text() == 'a':
                for marker in self.markers:
                    if (self.tmin <= marker.get_tmin() <= self.tmax or
                        self.tmin <= marker.get_tmax() <= self.tmax):
                        marker.set_selected(True)
                    
            elif key_event.text() == 'd':
                for marker in self.markers:
                    marker.set_selected(False)
                    
            elif key_event.text() in ('0', '1', '2', '3', '4', '5'):
                for marker in self.markers:
                    if marker.is_selected():
                        marker.set_kind(int(key_event.text()))
    
            elif key_event.key() == Qt.Key_Escape:
                if self.picking:
                    self.stop_picking(0,0,abort=True)
            
            elif key_event.key() == Qt.Key_PageDown:
                self.scroll_tracks(self.shown_tracks_range[1]-self.shown_tracks_range[0])
                
            elif key_event.key() == Qt.Key_PageUp:
                self.scroll_tracks(self.shown_tracks_range[0]-self.shown_tracks_range[1])
                
            elif key_event.text() == '+':
                self.zoom_tracks(0.,1.)
            
            elif key_event.text() == '-':
                self.zoom_tracks(0.,-1.)
                            
            self.update()
            self.update_status()
    
        def wheelEvent(self, wheel_event):
            amount = max(1.,abs(self.shown_tracks_range[0]-self.shown_tracks_range[1])/5.)
            
            if wheel_event.delta() < 0:
                wdelta = -amount
            else:
                wdelta = +amount
            
            trmin,trmax = self.track_to_screen.get_in_range()
            anchor = (self.track_to_screen.rev(wheel_event.y())-trmin)/(trmax-trmin)
            if wheel_event.modifiers() & Qt.ControlModifier:
                self.zoom_tracks( anchor, wdelta )
            else:
                self.scroll_tracks( -wdelta )
    
    
        def scroll_tracks(self, shift):
            shown = self.shown_tracks_range
            shiftmin = -shown[0]
            shiftmax = self.ntracks-shown[1]
            shift = max(shiftmin, shift)
            shift = min(shiftmax, shift)
            shown = shown[0] + shift, shown[1] + shift
            self.shown_tracks_range = int(shown[0]), int(shown[1])
            self.shown_tracks_start = self.shown_tracks_range[0]
            self.update()
            
        def zoom_tracks(self, anchor, delta):
            ntracks_shown = self.shown_tracks_range[1]-self.shown_tracks_range[0]
            ntracks_shown += int(round(delta))
            if not ( 1 <= ntracks_shown <= self.ntracks): return
            u = self.shown_tracks_start
            nu = max(0., u-anchor*delta)
            nv = nu + ntracks_shown
            if nv > self.ntracks:
                nu -= nv - self.ntracks
                nv -= nv - self.ntracks
            
            self.shown_tracks_start = nu
            self.shown_tracks_range = int(round(nu)), int(round(nv))
            
            self.update()       
    
        def printit(self):
            printer = QPrinter()
            printer.setOrientation(QPrinter.Landscape)
            
            dialog = QPrintDialog(printer, self)
            dialog.setWindowTitle('Print')
            
            if dialog.exec_() != QDialog.Accepted:
                return
            
            painter = QPainter()
            painter.begin(printer)
            page = printer.pageRect()
            self.drawit(painter, printmode=False, w=page.width(), h=page.height())
            painter.end()
            
            
        def savesvg(self):
            
            fn = QFileDialog.getSaveFileName(self, 
                'Save as SVG',
                os.path.join(os.environ['HOME'],  'untitled.svg'),
                'SVG (*.svg)'
            )
            
            generator = QSvgGenerator()
            generator.setFileName(fn)
            generator.setSize(QSize(842, 595))

            painter = QPainter()
            painter.begin(generator)
            self.drawit(painter, printmode=False, w=generator.size().width(), h=generator.size().height())
            painter.end()
            
        def paintEvent(self, paint_ev ):
            """Called by QT whenever widget needs to be painted"""
            painter = QPainter(self)
    
            if self.menuitem_antialias.isChecked():
                painter.setRenderHint( QPainter.Antialiasing )
                
            self.drawit( painter )
        
        def determine_box_styles(self):
            
            traces = list(self.pile.iter_traces())
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id))
            istyle = 0
            trace_styles = {}
            for itr, tr in enumerate(traces):
                if itr > 0:
                    other = traces[itr-1]
                    if not (other.nslc_id == tr.nslc_id and
                        other.deltat == tr.deltat and
                        abs(other.tmax - tr.tmin) < gap_lap_tolerance): istyle+=1
            
                trace_styles[tr.full_id, tr.deltat] = istyle
            
            self.trace_styles = trace_styles
            
        def draw_trace_boxes(self, p, time_projection, track_projections):
            
            for v_projection in track_projections.values():
                v_projection.set_in_range(0.,1.)
            
            selector = lambda x: x.overlaps(*time_projection.get_in_range())
            
            traces = list(self.pile.iter_traces(group_selector=selector, trace_selector=selector))
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id))
            istyle = 0
            for itr, tr in enumerate(traces):
                
                itrack = self.key_to_row[self.gather(tr)]
                if not itrack in track_projections: continue
                
                v_projection = track_projections[itrack]
                
                dtmin = time_projection(tr.tmin)
                dtmax = time_projection(tr.tmax)
                
                dvmin = v_projection(0.)
                dvmax = v_projection(1.)
            
                istyle = self.trace_styles.get((tr.full_id, tr.deltat), 0)
                style = box_styles[istyle%len(box_styles)]
                rect = QRectF( dtmin, dvmin, dtmax-dtmin, dvmax-dvmin )
                p.fillRect(rect, style.fill_brush)
                
                p.setPen(style.frame_pen)
                p.drawRect(rect)
            
        def drawit(self, p, printmode=False, w=None, h=None):
            """This performs the actual drawing."""
    
            if h is None: h = self.height()
            if w is None: w = self.width()
            
            if printmode:
                primary_color = (0,0,0)
            else:
                primary_color = pyrocko.plot.tango_colors['aluminium5']
            
            ax_h = self.ax_height
            
            vbottom_ax_projection = Projection()
            vtop_ax_projection = Projection()
            vcenter_projection = Projection()
            
            self.time_projection.set_out_range(0, w)
            vbottom_ax_projection.set_out_range(h-ax_h, h)
            vtop_ax_projection.set_out_range(0, ax_h)
            vcenter_projection.set_out_range(ax_h, h-ax_h)
            vcenter_projection.set_in_range(0.,1.)
            self.track_to_screen.set_out_range(ax_h, h-ax_h)
            
            ntracks = self.ntracks
            self.track_to_screen.set_in_range(*self.shown_tracks_range)
            track_projections = {}
            for i in range(*self.shown_tracks_range):
                proj = Projection()
                proj.set_out_range(self.track_to_screen(i+0.05),self.track_to_screen(i+1.-0.05))
                track_projections[i] = proj
            
                    
            if self.tmin < self.tmax:
                self.time_projection.set_in_range(self.tmin, self.tmax)
                vbottom_ax_projection.set_in_range(0, ax_h)
    
                self.tax.drawit( p, self.time_projection, vbottom_ax_projection )
                
                yscaler = pyrocko.plot.AutoScaler()
                if not printmode and self.menuitem_showboxes.isChecked():
                    self.draw_trace_boxes(p, self.time_projection, track_projections)
                
                if self.floating_marker:
                    self.floating_marker.draw(p, self.time_projection, vcenter_projection)
                
                for marker in self.markers:
                    if marker.get_tmin() < self.tmax and self.tmin < marker.get_tmax():
                        marker.draw(p, self.time_projection, vcenter_projection)
                    
                primary_pen = QPen(QColor(*primary_color))
                p.setPen(primary_pen)
                            
                processed_traces = self.prepare_cutout(self.tmin, self.tmax, 
                                                    trace_selector=self.trace_selector, 
                                                    degap=self.menuitem_degap.isChecked())
                
                color_lookup = dict([ (k,i) for (i,k) in enumerate(self.color_keys) ])
                
                self.track_to_nslc_ids = {}
                min_max_for_annot = {}
                if processed_traces:
                    yscaler = pyrocko.plot.AutoScaler()
                    data_ranges = pyrocko.trace.minmax(processed_traces, key=self.scaling_key, mode=self.scalingbase)
                    for trace in processed_traces:
                        itrack = self.key_to_row[self.gather(trace)]
                        if itrack in track_projections:
                            if itrack not in self.track_to_nslc_ids:
                                self.track_to_nslc_ids[itrack] = set()
                            self.track_to_nslc_ids[itrack].add( trace.nslc_id )
                            
                            track_projection = track_projections[itrack]
                            data_range = data_ranges[self.scaling_key(trace)]
                            ymin, ymax, yinc = yscaler.make_scale( data_range )
                            track_projection.set_in_range(ymax,ymin)
        
                            udata = self.time_projection(trace.get_xdata())
                            vdata = track_projection( self.gain*trace.get_ydata() )
                            
                            umin, umax = self.time_projection.get_out_range()
                            vmin, vmax = track_projection.get_out_range()
                            
                            trackrect = QRectF(umin,vmin, umax-umin, vmax-vmin)
                        
                            qpoints = make_QPolygonF( udata, vdata )
                                
                            if self.menuitem_cliptraces.isChecked(): p.setClipRect(trackrect)
                            if self.menuitem_colortraces.isChecked():
                                color = pyrocko.plot.color(color_lookup[self.color_gather(trace)])
                                pen = QPen(QColor(*color))
                                p.setPen(pen)
                            
                            p.drawPolyline( qpoints )
                            
                            if self.floating_marker:
                                self.floating_marker.draw_trace(p, trace, self.time_projection, track_projection, self.gain)
                                
                            for marker in self.markers:
                                if marker.get_tmin() < self.tmax and self.tmin < marker.get_tmax():
                                    marker.draw_trace(p, trace, self.time_projection, track_projection, self.gain)
                            p.setPen(primary_pen)
                                
                            if self.menuitem_cliptraces.isChecked(): p.setClipRect(0,0,w,h)
                            
                            if  itrack not in min_max_for_annot:
                                min_max_for_annot[itrack] = (ymin, ymax)
                            else:
                                if min_max_for_annot is not None and min_max_for_annot[itrack] != (ymin, ymax):
                                    min_max_for_annot[itrack] = None
                            
                p.setPen(primary_pen)
                                                        
                font = QFont()
                font.setBold(True)
                p.setFont(font)
                fm = p.fontMetrics()
                label_bg = QBrush( QColor(255,255,255,100) )
                
                for key in self.track_keys:
                    itrack = self.key_to_row[key]
                    if itrack in track_projections:
                        plabel = ' '.join([ x for x in key if x])
                        label = QString( plabel)
                        rect = fm.boundingRect( label )
                        
                        lx = 10
                        ly = self.track_to_screen(itrack+0.5)
                        
                        rect.translate( lx, ly )
                        p.fillRect( rect, label_bg )
                        p.drawText( lx, ly, label )
                        
                        if (self.menuitem_showscalerange.isChecked() and itrack in min_max_for_annot):
                            if min_max_for_annot[itrack] is not None:
                                plabel = '(%.2g, %.2g)' % min_max_for_annot[itrack]
                            else:
                                plabel = 'Mixed Scales!'
                            label = QString( plabel)
                            rect = fm.boundingRect( label )
                            lx = w-10-rect.width()
                            rect.translate( lx, ly )
                            p.fillRect( rect, label_bg )
                            p.drawText( lx, ly, label )
    
        def prepare_cutout(self, tmin, tmax, trace_selector=None, degap=True):
                    
            fft_filtering = self.menuitem_fft_filtering.isChecked()
            lphp = self.menuitem_lphp.isChecked()
            
            vec = (tmin, tmax, trace_selector, degap, self.lowpass, self.highpass, fft_filtering, lphp,
                self.min_deltat, self.rotate, self.shown_tracks_range,
                self.menuitem_allowdownsampling.isChecked(), self.pile.get_update_count())
                
            if vec == self.old_vec and not (self.reloaded or self.menuitem_watch.isChecked()):
                return self.old_processed_traces
            
            self.old_vec = vec
            
            if self.lowpass is not None:
                deltat_target = 1./self.lowpass * 0.25
                ndecimate = min(50, max(1, int(round(deltat_target / self.min_deltat))))
                tpad = 1./self.lowpass * 2.
            else:
                ndecimate = 1
                tpad = self.min_deltat*5.
                
            if self.highpass is not None:
                tpad = max(1./self.highpass * 2., tpad)
            
            tpad = min(tmax-tmin, tpad)
            tpad = max(self.min_deltat*5., tpad)
                
            nsee_points_per_trace = 5000*10
            see_data_range = ndecimate*nsee_points_per_trace*self.min_deltat
            processed_traces = []
            
            if (tmax - tmin) < see_data_range:
                            
                for traces in self.pile.chopper( tmin=tmin, tmax=tmax, tpad=tpad,
                                                want_incomplete=True,
                                                degap=degap,
                                                keep_current_files_open=True, trace_selector=trace_selector ):
                    for trace in traces:
                        
                        if fft_filtering:
                            if self.lowpass is not None or self.highpass is not None:
                                high, low = 1./(trace.deltat*len(trace.ydata)),  1./(2.*trace.deltat)
                                
                                if self.lowpass is not None:
                                    low = self.lowpass
                                if self.highpass is not None:
                                    high = self.highpass
                                    
                                trace.bandpass_fft(high, low)
                            
                        else:
                            if self.lowpass is not None:
                                deltat_target = 1./self.lowpass * 0.2
                                ndecimate = max(1, int(math.floor(deltat_target / trace.deltat)))
                                ndecimate2 = int(math.log(ndecimate,2))
                                
                            else:
                                ndecimate = 1
                                ndecimate2 = 0
                            
                            if ndecimate2 > 0 and self.menuitem_allowdownsampling.isChecked():
                                for i in range(ndecimate2):
                                    trace.downsample(2)
                            
                            
                            if not lphp and (self.lowpass is not None and self.highpass is not None and
                                self.lowpass < 0.5/trace.deltat and
                                self.highpass < 0.5/trace.deltat and
                                self.highpass < self.lowpass):
                                trace.bandpass(2,self.highpass, self.lowpass)
                            else:
                                if self.lowpass is not None:
                                    if self.lowpass < 0.5/trace.deltat:
                                        trace.lowpass(4,self.lowpass)
                                
                                if self.highpass is not None:
                                    if self.lowpass is None or self.highpass < self.lowpass:
                                        if self.highpass < 0.5/trace.deltat:
                                            trace.highpass(4,self.highpass)
                        try:
                            trace = trace.chop(tmin-trace.deltat*4.,tmax+trace.deltat*4.)
                        except pyrocko.trace.NoData:
                            continue
                            
                        if len(trace.get_ydata()) < 2: continue
                        
                        processed_traces.append(trace)
                
            if self.rotate != 0.0:
                phi = self.rotate/180.*math.pi
                cphi = math.cos(phi)
                sphi = math.sin(phi)
                for a in processed_traces:
                    for b in processed_traces: 
                        if (a.network == b.network and a.station == b.station and a.location == b.location and
                            a.channel.lower().endswith('n') and b.channel.lower().endswith('e') and
                            abs(a.deltat-b.deltat) < a.deltat*0.001 and abs(a.tmin-b.tmin) < a.deltat*0.01 and
                            len(a.get_ydata()) == len(b.get_ydata())):
                            
                            aydata = a.get_ydata()*cphi+b.get_ydata()*sphi
                            bydata =-a.get_ydata()*sphi+b.get_ydata()*cphi
                            a.set_ydata(aydata)
                            b.set_ydata(bydata)
                            
            self.old_processed_traces = processed_traces
            return processed_traces
        
        def scalingbase_change(self, ignore):
            for menuitem, scalingbase in self.menuitems_scalingbase:
                if menuitem.isChecked():
                    self.scalingbase = scalingbase
        
        def scalingmode_change(self, ignore):
            for menuitem, scaling_key in self.menuitems_scaling:
                if menuitem.isChecked():
                    self.scaling_key = scaling_key
    
        def sortingmode_change(self, ignore=None):
            for menuitem, (gather, order, color) in self.menuitems_sorting:
                if menuitem.isChecked():
                    self.set_gathering(gather, order, color)
    
        def lowpass_change(self, value, ignore):
            if num.isfinite(value):
                self.lowpass = value
            else:
                self.lowpass = None
            self.passband_check()
            self.update()
            
        def highpass_change(self, value, ignore):
            if num.isfinite(value):
                self.highpass = value
            else:
                self.highpass = None
            self.passband_check()
            self.update()
    
        def passband_check(self):
            if self.highpass and self.lowpass and self.highpass >= self.lowpass:
                self.message = 'Corner frequency of highpass larger than corner frequency of lowpass! I will now deactivate the higpass.'
                self.update_status()
            else:
                oldmess = self.message
                self.message = None
                if oldmess is not None:
                    self.update_status()        
        
        def gain_change(self, value, ignore):
            self.gain = value
            self.update()
            
        def rot_change(self, value, ignore):
            self.rotate = value
            self.update()
    
        def get_min_deltat(self):
            return self.min_deltat
        
        def deselect_all(self):
            for marker in self.markers:
                marker.set_selected(False)
        
        def animate_picking(self):
            point = self.mapFromGlobal(QCursor.pos())
            self.update_picking(point.x(), point.y(), doshift=True)
            
        def get_nslc_ids_for_track(self, ftrack):
            itrack = int(ftrack)
            if itrack in self.track_to_nslc_ids:
                return self.track_to_nslc_ids[int(ftrack)]
            else:
                return []
                
        def stop_picking(self, x,y, abort=False):
            if self.picking:
                self.update_picking(x,y, doshift=False)
                #self.picking.hide()
                self.picking = None
                self.picking_down = None
                self.picking_timer.stop()
                self.picking_timer = None
                if not abort:
                    tmi = self.floating_marker.tmin
                    tma = self.floating_marker.tmax
                    self.markers.append(self.floating_marker)
                    self.floating_marker.set_selected(True)
                    print self.floating_marker
                
                self.floating_marker = None
        
        
        def start_picking(self, ignore):
            
            if not self.picking:
                self.deselect_all()
                self.picking = QRubberBand(QRubberBand.Rectangle)
                point = self.mapFromGlobal(QCursor.pos())
                
                gpoint = self.mapToGlobal( QPoint(point.x(), 0) )
                self.picking.setGeometry( gpoint.x(), gpoint.y(), 1, self.height())
                t = self.time_projection.rev(point.x())
                
                ftrack = self.track_to_screen.rev(point.y())
                nslc_ids = self.get_nslc_ids_for_track(ftrack)
                self.floating_marker = Marker(nslc_ids, t,t)
                self.floating_marker.set_selected(True)
    
                ##self.picking.show()
                #self.setMouseTracking(True)
                
                self.picking_timer = QTimer()
                self.connect( self.picking_timer, SIGNAL("timeout()"), self.animate_picking )
                self.picking_timer.setInterval(50)
                self.picking_timer.start()
    
        
        def update_picking(self, x,y, doshift=False):
            if self.picking:
                mouset = self.time_projection.rev(x)
                dt = 0.0
                if mouset < self.tmin or mouset > self.tmax:
                    if mouset < self.tmin:
                        dt = -(self.tmin - mouset)
                    else:
                        dt = mouset - self.tmax 
                    ddt = self.tmax-self.tmin
                    dt = max(dt,-ddt/10.)
                    dt = min(dt,ddt/10.)
                    
                x0 = x
                if self.picking_down is not None:
                    x0 = self.time_projection(self.picking_down[0])
                
                w = abs(x-x0)
                x0 = min(x0,x)
                
                tmin, tmax = self.time_projection.rev(x0), self.time_projection.rev(x0+w)
                tmin, tmax = ( max(working_system_time_range[0], tmin),
                            min(working_system_time_range[1], tmax))
                                
                p1 = self.mapToGlobal( QPoint(x0, 0))
                
                self.picking.setGeometry( p1.x(), p1.y(), max(w,1), self.height())
                
                ftrack = self.track_to_screen.rev(y)
                nslc_ids = self.get_nslc_ids_for_track(ftrack)
                self.floating_marker.set(nslc_ids, tmin, tmax)
                
                if dt != 0.0 and doshift:
                    self.set_time_range(self.tmin+dt, self.tmax+dt)
                
                self.update()
    
        def update_status(self):
            
            if self.message is None:
                point = self.mapFromGlobal(QCursor.pos())
                
                mouse_t = self.time_projection.rev(point.x())
                if not is_working_time(mouse_t): return
                if self.floating_marker:
                    tmi, tma = self.floating_marker.tmin, self.floating_marker.tmax
                    tt, ms = gmtime_x(tmi)
                
                    if tmi == tma:
                        message = mystrftime(fmt='Pick: %Y-%m-%d %H:%M:%S .%r', tt=tt, milliseconds=ms)
                    else:
                        srange = '%g s' % (tma-tmi)
                        message = mystrftime(fmt='Start: %Y-%m-%d %H:%M:%S .%r Length: '+srange, tt=tt, milliseconds=ms)
                else:
                    tt, ms = gmtime_x(mouse_t)
                
                    message = mystrftime(fmt=None,tt=tt,milliseconds=ms)
            else:
                message = self.message
                
            sb = self.window().statusBar()
            sb.clearMessage()
            sb.showMessage(message)
            
        def myclose(self):
            self.window().close()
            
            
        def fuck(self):
            import pysacio
            
            processed_traces = self.prepare_cutout(self.tmin,self.tmax)
            sacdir = tempfile.mkdtemp(prefix='HERE_LIVES_SAC_')
            os.chdir(sacdir)
            
            sys.stderr.write('\n\n --> Dumping SAC files to %s  <--\n\n\n' % sacdir)
            
            for trace in processed_traces:
                # FIXME:
                sactr = pysacio.from_mseed_trace(trace)
                sactr.write('trace-%s-%s-%s-%s.sac' % trace.nslc_id)
            
            self.myclose()
            Global.sacflag = True
            
            
    return PileOverview

PileOverview = MakePileOverviewClass(QWidget)
GLPileOverview = MakePileOverviewClass(QGLWidget)
        
class MyValueEdit(QLineEdit):

    def __init__(self, *args):
        apply(QLineEdit.__init__, (self,) + args)
        self.value = 0.
        self.mi = 0.
        self.ma = 1.
        self.connect( self, SIGNAL("editingFinished()"), self.myEditingFinished )
        self.err_palette = QPalette()
        self.err_palette.setColor( QPalette.Base, QColor(255,200,200) )
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
            self.setPalette( self.err_palette )
        
        self.lock = False
        
    def adjust_text(self):
        self.setText( ("%8.5g" % self.value).strip() )
        
class ValControl(QFrame):

    def __init__(self, low_is_none=False, high_is_none=False, *args):
        apply(QFrame.__init__, (self,) + args)
        self.layout = QHBoxLayout( self )
        #self.layout.setSpacing(5)
        self.lname = QLabel( "name", self )
        self.lname.setFixedWidth(120)
        self.lvalue = MyValueEdit( self )
        self.lvalue.setFixedWidth(100)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMaximum( 10000 )
        self.slider.setSingleStep( 100 )
        self.slider.setPageStep( 1000 )
        self.slider.setTickPosition( QSlider.NoTicks )
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
        self.lvalue.setValue( self.cur )
        self.slider.setValue( self.cursl )
        self.mute = False
        
    def get_value(self):
        return self.cur
        
    def slided(self,val):
        if self.cursl != val:
            self.cursl = val
            self.cur = self.s2v(self.cursl)
            self.lvalue.setValue( self.cur )
            self.fire_valchange()
            
    def edited(self,val):
        if self.cur != val:
            self.cur = val
            cursl = self.v2s(val)
            if (cursl != self.cursl):
                self.slider.setValue( cursl )
            
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

class PileViewer(QFrame):
    '''PileOverview + Controls'''
    
    def __init__(self, pile, ntracks_shown_max, use_opengl=False, *args):
        apply(QFrame.__init__, (self,) + args)
        
        if use_opengl:
            self.pile_overview = GLPileOverview(pile, ntracks_shown_max=ntracks_shown_max)
        else:
            self.pile_overview = PileOverview(pile, ntracks_shown_max=ntracks_shown_max)
        
        layout = QGridLayout()
        self.setLayout( layout )
        
        layout.addWidget( self.pile_overview, 0, 0 )
        
        minfreq = 0.001
        maxfreq = 0.5/self.pile_overview.get_min_deltat()
        if maxfreq < 100.*minfreq:
            minfreq = maxfreq*0.00001
        
        self.lowpass_widget = ValControl(high_is_none=True)
        self.lowpass_widget.setup('Lowpass [Hz]:', minfreq, maxfreq, maxfreq, 0)
        self.highpass_widget = ValControl(low_is_none=True)
        self.highpass_widget.setup('Highpass [Hz]:', minfreq, maxfreq, minfreq, 1)
        self.gain_widget = ValControl()
        self.gain_widget.setup('Gain', 0.001, 1000., 1., 2)
        self.rot_widget = LinValControl()
        self.rot_widget.setup('Rotate', -180., 180., 0., 3)
        self.connect( self.lowpass_widget, SIGNAL("valchange(float,int)"), self.pile_overview.lowpass_change )
        self.connect( self.highpass_widget, SIGNAL("valchange(float,int)"), self.pile_overview.highpass_change )
        self.connect( self.gain_widget, SIGNAL("valchange(float,int)"), self.pile_overview.gain_change )
        self.connect( self.rot_widget, SIGNAL("valchange(float,int)"), self.pile_overview.rot_change )
        
        layout.addWidget( self.highpass_widget, 1,0 )
        layout.addWidget( self.lowpass_widget, 2,0 )
        layout.addWidget( self.gain_widget, 3,0 )
        layout.addWidget( self.rot_widget, 4,0 )
    
    def get_pile(self):
        return self.pile_overview.get_pile()

from forked import Forked
class SnufflerOnDemand(QApplication, Forked):
    def __init__(self, *args):
        apply(QApplication.__init__, (self,) + args)
        Forked.__init__(self, flipped=True)
        self.timer = QTimer( self )
        self.connect( self.timer, SIGNAL("timeout()"), self.periodical ) 
        self.timer.setInterval(1000)
        self.timer.start()
        self.caller_has_quit = False
        self.viewers = []
        self.windows = []
        pile = pyrocko.pile.Pile()
        self.new_viewer(pile)
        
    def dispatch(self, command, args, kwargs):
        method = getattr(self, command)
        method(*args, **kwargs)
        
    def add_traces(self, traces, iviewer=0):
        pile = self.viewers[iviewer].get_pile()
        memfile = pyrocko.pile.MemTracesFile(None, traces)
        pile.add_file(memfile)
        
    def periodical(self):
        if not self.caller_has_quit:
            self.caller_has_quit = not self.process()
            
    def new_viewer(self, pile, ntracks_shown_max=20):
        pile_viewer = PileViewer(pile, ntracks_shown_max=ntracks_shown_max)
        win = QMainWindow()
        win.setCentralWidget(pile_viewer)
        win.setWindowTitle( "Snuffler %i" % (len(self.viewers)+1) )        
        win.show()
        self.viewers.append(pile_viewer)
        self.windows.append(win)
        
    def run(self):
        self.exec_()
    

def snuffle(traces=None, filenames=None, pile=None):
    pile = pyrocko.pile.Pile()    
    
    if Global.appOnDemand is None:
        app = Global.appOnDemand = SnufflerOnDemand([])
        
    app = Global.appOnDemand
    if traces is not None:
        print app.call('add_traces', traces)
    
    
def sac_exec():
    import readline, subprocess, atexit
    sac = subprocess.Popen(['sac'], stdin=subprocess.PIPE)
    sac.stdin.write('read *.sac\n')
    sac.stdin.flush()
    
    histfile = os.path.join(os.environ["HOME"], ".snuffler_sac_history")
    try:
        readline.read_history_file(histfile)
    except IOError:
        pass
    
    atexit.register(readline.write_history_file, histfile)
    
    while True:
        try:
            s = raw_input()
        except EOFError:
            break
        
        sac.stdin.write(s+"\n")
        if s in ('q', 'quit', 'exit'): break
        sac.stdin.flush()
        
    sac.stdin.close()
