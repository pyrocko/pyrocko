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
import traceback
from optparse import OptionParser

import pyrocko.model
import pyrocko.pile
import pyrocko.shadow_pile
import pyrocko.trace
import pyrocko.util
import pyrocko.plot
import pyrocko.snuffling
import pyrocko.snufflings
from pyrocko.nano import Nano
from pyrocko.gui_util import ValControl, LinValControl

from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import *
from PyQt4.QtSvg import *

logger = logging.getLogger('pyrocko.pile_viewer')

class Global:
    sacflag = False
    appOnDemand = None

class m_float(float):
    
    def __str__(self):
        if abs(self) >= 10000.:
            return '%g km' % round(self/1000.,0)
        elif abs(self) >= 1000.:
            return '%g km' % round(self/1000.,1)
        else:
            return '%.5g m' % self
        
class deg_float(float):
    
    def __str__(self):
        return '%4.0f' % self
 
class sector_int(int):
    
    def __str__(self):
        return '[%i]' % self
 
gap_lap_tolerance = 5.

class Timer:
    def __init__(self):
        self._start = None
        self._stop = None
    
    def start(self):
        self._start = os.times()
    
    def stop(self):
        self._stop = os.times()

    def get(self):
        a = self._start
        b = self._stop
        if a is not None and b is not None:
            return tuple([ b[i] - a[i] for i in xrange(5) ])
        else:
            return tuple([ 0. ] * 5)
        
    def __sub__(self, other):
        a = self.get()
        b = other.get()
        return tuple( [ a[i] - b[i] for i in xrange(5) ] )

class Integrator(pyrocko.shadow_pile.ShadowPile):

    def process(self, iblock, tmin, tmax, traces):
        for trace in traces:
            trace.ydata -= trace.ydata.mean()
            trace.ydata = num.cumsum(trace.ydata)
        
        return traces
        
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

def draw_label( p, x,y, label_str, label_bg, anchor='BL' ):
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
    p.fillRect( rect, label_bg )
    p.drawText( tx, ty, label )

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
    if inc < 0.001:
        l0_fmt = '.%u'
        l0_center = False
        l1_fmt = '%H:%M:%S'
        l1_trig = 6
        l2_fmt = '%b %d, %Y'
        l2_trig = 3
    elif inc < 1:
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
   tt = time.gmtime(int(timestamp))
   tts = tt[0:3] + (0,0,0) + tt[6:9]
   return calendar.timegm(tts)

def month_start(timestamp):
   tt = time.gmtime(int(timestamp))
   tts = tt[0:2] + (1,0,0,0) + tt[6:9]
   return calendar.timegm(tts)

def year_start(timestamp):
    tt = time.gmtime(int(timestamp))
    tts = tt[0:1] + (1,1,0,0,0) + tt[6:9]
    return calendar.timegm(tts)

def gmtime_x(timestamp):
    if isinstance(timestamp, Nano):
        etimestamp = int(timestamp)
    else:
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
            base = mi_day+math.ceil(float(mi-mi_day)/inc)*inc
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
    fmt3 = fmt2.replace('%u', '%06i' % int(round(milliseconds*1000)))
    return time.strftime(fmt3, tt)
        
        
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
        if isinstance(xmin, Nano) or isinstance(xmax, Nano):
            self.xr = xmin, xmax
        else:
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
    
    def clipped(self, x):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return min(umax, max(umin, umin + (x-xmin)*((umax-umin)/(xmax-xmin))))
        
    def rev(self, u):
        umin, umax = self.ur
        xmin, xmax = self.xr
        return xmin + (u-umin)*((xmax-xmin)/(umax-umin))

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
    def load_markers(fn):
        markers = []
        f = open(fn, 'r')
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
            
            vdata = track_projection( gain*snippet.get_ydata() )
            udata_min = float(time_projection(snippet.tmin))
            udata_max = float(time_projection(snippet.tmin+snippet.deltat*(vdata.size-1)))
            udata = num.linspace(udata_min, udata_max, vdata.size)
            
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

        label = self.get_label()
        if label:
            label_bg = QBrush( QColor(220,220,220) )
            u = time_projection(self.tmin)
            v0, v1 = track_projection.get_out_range()
            draw_label( p, u-5., v0, label, label_bg, 'TR')

        try: drawpoint(self.tmin, trace.interpolate(self.tmin))
        except IndexError: pass
        try: drawpoint(self.tmax, trace.interpolate(self.tmax))
        except IndexError: pass            
    
    def get_label(self):
        return None 

class EventMarker(Marker):
    def __init__(self, event):
        Marker.__init__(self, [], event.time, event.time)
        self._event = event

    def draw(self, p, time_projection, y_projection):
        Marker.draw(self, p, time_projection, y_projection)
        
        u = time_projection(self.tmin)
        v0, v1 = y_projection.get_out_range()
        t = []
        mag = self._event.magnitude
        if mag is not None:
            t.append('M%3.1f' % mag)
        
        reg = self._event.region
        if reg is not None:
            t.append(reg)
        
        label = ' '.join(t)
        label_bg = QBrush( QColor(220,220,220) )
        draw_label( p, u, v0-10., label, label_bg, 'CB')

    def get_event(self):
        return self._event

    def draw_trace(self, p, trace, time_projection, track_projection, gain):
        pass

class PhaseMarker(Marker):

    def __init__(self, nslc_ids, tmin, tmax, event=None, phasename=None, polarity=None, automatic=None):
        Marker.__init__(self, nslc_ids, tmin, tmax)
        self._event = event
        self._phasename = phasename
        self._polarity = polarity
        self._automatic = automatic

    def get_label(self):
        t = []
        if self._phasename is not None:
            t.append(self._phasename)
        if self._polarity is not None:
            t.append(self._polarity)

        if self._automatic:
            t.append('@')

        return ''.join(t)


class PileOverviewException(Exception):
    pass

def MakePileOverviewClass(base):
    
    class PileOverview(base):

        def __init__(self, pile, ntracks_shown_max, add_panel_hook, *args):
            if base == QGLWidget:
                apply(base.__init__, (self, QGLFormat(QGL.SampleBuffers)) + args)
            else:
                apply(base.__init__, (self,) + args)

            self.pile = pile
            self.ax_height = 80
            self.add_panel_hook = add_panel_hook
            
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
            self.pile_has_changed = False
            

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

            self.menuitem_setorigin = QAction('Set Origin', self.menu)
            self.menu.addAction(self.menuitem_setorigin)
            self.connect( self.menuitem_setorigin, SIGNAL("triggered(bool)"), self.set_event_marker_as_origin)
 
            self.menu.addSeparator()

            menudef = [
                ('Sort by Names',
                    lambda tr: () ),
                ('Sort by Distance',
                    lambda tr: self.station_attrib(tr, lambda sta: (m_float(sta.dist_m),), lambda tr: (None,) )),
                ('Sort by Azimuth',
                    lambda tr: self.station_attrib(tr, lambda sta: (deg_float(sta.azimuth),), lambda tr: (None,) )),
                ('Sort by Distance in 12 Azimuthal Blocks',
                    lambda tr: self.station_attrib(tr, lambda sta: (sector_int(round((sta.azimuth+15.)/30.)), m_float(sta.dist_m)),
                                                       lambda tr: (None,None) )),
            ]
            self.menuitems_ssorting = add_radiobuttongroup(self.menu, menudef, self, self.s_sortingmode_change)
            
            self._ssort = lambda tr: ()
            
            self.menu.addSeparator()
            
            menudef = [
                ('Subsort by Network, Station, Location, Channel', 
                    ( lambda tr: self.ssort(tr) + tr.nslc_id,     # gathering
                    lambda a,b: cmp(a,b),      # sorting
                    lambda tr: tr.location )),  # coloring
                ('Subsort by Network, Station, Channel, Location', 
                    ( lambda tr: self.ssort(tr) + (tr.network, tr.station, tr.channel, tr.location),
                    lambda a,b: cmp(a,b),
                    lambda tr: tr.channel )),
                ('Subsort by Station, Network, Channel, Location', 
                    ( lambda tr: self.ssort(tr) + (tr.station, tr.network, tr.channel, tr.location),
                    lambda a,b: cmp(a,b),
                    lambda tr: tr.channel )),
                ('Subsort by Location, Network, Station, Channel', 
                    ( lambda tr: self.ssort(tr) + (tr.location, tr.network, tr.station, tr.channel),
                    lambda a,b: cmp(a,b),
                    lambda tr: tr.channel )),
                ('Subsort by Network, Station, Channel (Grouped by Location)',
                    ( lambda tr: self.ssort(tr) + (tr.network, tr.station, tr.channel),
                    lambda a,b: cmp(a,b),
                    lambda tr: tr.location )),
                ('Subsort by Station, Network, Channel (Grouped by Location)',
                    ( lambda tr: self.ssort(tr) + (tr.station, tr.network, tr.channel),
                    lambda a,b: cmp(a,b),
                    lambda tr: tr.location )),
                
            ]
            self.menuitems_sorting = add_radiobuttongroup(self.menu, menudef, self, self.sortingmode_change)
            
            self.menu.addSeparator()
            
            self.menuitem_antialias = QAction('Antialiasing', self.menu)
            self.menuitem_antialias.setCheckable(True)
            self.menu.addAction(self.menuitem_antialias)
            
            self.menuitem_liberal_fetch = QAction('Liberal Fetch Optimization', self.menu)
            self.menuitem_liberal_fetch.setCheckable(True)
            self.menu.addAction(self.menuitem_liberal_fetch)
            
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
            
            self.menuitem_fixscalerange = QAction('Fix Scale Range', self.menu)
            self.menuitem_fixscalerange.setCheckable(True)
            self.menu.addAction(self.menuitem_fixscalerange)
                
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

            self.menuitem_test = QAction('Test', self.menu)
            self.menuitem_test.setCheckable(True)
            self.menuitem_test.setChecked(False)
            self.menu.addAction(self.menuitem_test)
            self.connect( self.menuitem_test, SIGNAL("toggled(bool)"), self.toggletest )

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
            self.set_time_range(self.pile.get_tmin(), self.pile.get_tmax())
            self.time_projection.set_out_range(0., self.width())
                
            self.gather = None
    
            self.trace_filter = None
            self.quick_filter = None
            self.quick_filter_pattern = None, None
            self.blacklist = []
            
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
            self.snuffling_modules = {}
            self.snuffling_paths = [ os.path.join(user_home_dir, '.snufflings') ]
            self.default_snufflings = None
            self.setup_snufflings()
            
            self.stations = {}
            
            self.timer_draw = Timer()
            self.timer_cutout = Timer()
            
            self.interactive_range_change_time = 0.0
            self.interactive_range_change_delay_time = 10.0
            self.follow_timer = None
            
            self.sortingmode_change_time = 0.0
            self.sortingmode_change_delay_time = None
            
            self.old_data_ranges = {}
            
            self.error_messages = {}
            
        def fail(self, reason):
            box = QMessageBox(self)
            box.setText(reason)
            box.exec_()
    
        def set_trace_filter(self, filter_func):
            self.trace_filter = filter_func
            self.sortingmode_change()

        def update_trace_filter(self):
            if self.blacklist:
                blacklist_func = lambda tr: not pyrocko.util.match_nslc(self.blacklist, tr.nslc_id)
            else:
                blacklist_func = None
            
            if self.quick_filter is None and blacklist_func is None:
                self.set_trace_filter( None )
            elif self.quick_filter is None:
                self.set_trace_filter( blacklist_func )
            elif blacklist_func is None:
                self.set_trace_filter( self.quick_filter )                
            else:
                self.set_trace_filter( lambda tr: blacklist_func(tr) and self.quick_filter(tr) )

        def set_quick_filter(self, filter_func):
            self.quick_filter = filter_func
            self.update_trace_filter()
            
        def set_quick_filter_pattern(self, pattern, inputline=None):
            if pattern is not None:
                self.set_quick_filter(lambda tr: pyrocko.util.match_nslc(pattern, tr.nslc_id))
            else:
                self.set_quick_filter(None)
                
            self.quick_filter_pattern = pattern, inputline
            
        def get_quick_filter_pattern(self):
            return self.quick_filter_pattern
            
        def add_blacklist_pattern(self, pattern):
            if pattern in self.blacklist:
                self.blacklist.remove(pattern)
            self.blacklist.append(pattern)
            
            logger.info('Blacklist is [ %s ]' % ', '.join(self.blacklist))
            self.update_trace_filter()
            
        def remove_blacklist_pattern(self, pattern):
            if pattern in self.blacklist:
                self.blacklist.remove(pattern)
            else:
                raise PileOverviewException('Pattern not found in blacklist.')
            
            logger.info('Blacklist is [ %s ]' % ', '.join(self.blacklist))
            self.update_trace_filter()
            
        def clear_blacklist(self):
            self.blacklist = []
            self.update_trace_filter()
            
        def ssort(self, tr):
            return self._ssort(tr)
        
        def station_key(self, x):
            return x.network, x.station
        
        def station_attrib(self, tr, getter, default_getter):
            sk = self.station_key(tr)
            if sk in self.stations:
                station = self.stations[sk]
                return getter(station)
            else:
                return default_getter(tr)
            
        def set_stations(self, stations):
            self.stations = {}
            self.add_stations(stations)
        
        def add_stations(self, stations):
            for station in stations:
                sk = self.station_key(station)
                self.stations[sk] = station
        
        def add_event(self, event):
            marker = EventMarker(event)
            self.add_marker( marker )
       
        def set_event_marker_as_origin(self, ignore):
            selected = self.selected_markers()
            if not selected:
                self.fail('An event marker must be selected.')
                return

            m = selected[0]
            if not isinstance(m, EventMarker):
                self.fail('Selected marker is not an event.')
                return

            location = m.get_event()
            self.set_origin(location)

        def set_origin(self, location):
            for station in self.stations.values():
                station.set_event_relative_data(location)
            self.sortingmode_change()
        
        def toggletest(self, checked):
            if checked:
                sp = Integrator()
                
                self.add_shadow_pile(sp)
            else:
                self.remove_shadow_piles()
        
        def add_shadow_pile(self, shadow_pile):
            shadow_pile.set_basepile(self.pile)
            shadow_pile.add_listener(self)
            self.pile = shadow_pile
        
        def remove_shadow_piles(self):
            self.pile = self.pile.get_basepile()
            
        def iter_snuffling_modules(self):
            for path in self.snuffling_paths:
                
                if not os.path.isdir(path): 
                    continue
                
                for fn in os.listdir(path):
                    if not fn.endswith('.py'):
                        continue
                    
                    name = fn[:-3]
                    if (path, name) not in self.snuffling_modules:
                        self.snuffling_modules[path, name] = \
                            pyrocko.snuffling.SnufflingModule(path, name, self)
                    
                    yield self.snuffling_modules[path, name]
                    
        def setup_snufflings(self):
            # user snufflings
            for mod in self.iter_snuffling_modules():
                try:
                    mod.load_if_needed()
                except pyrocko.snuffling.BrokenSnufflingModule, e:
                    logger.warn( 'Snuffling module "%s" is broken' % e )

            # load the default snufflings on first run
            if self.default_snufflings is None:
                self.default_snufflings = pyrocko.snufflings.__snufflings__()
                for snuffling in self.default_snufflings:
                    self.add_snuffling(snuffling)
                
        def add_snuffling(self, snuffling):
            snuffling.init_gui(self, self, self.add_panel_hook, self.snufflings_menu, self.add_snuffling_menuitem)
            self.update()
            
        def remove_snuffling(self, snuffling):
            snuffling.delete_gui()
            self.update()
            
        def add_snuffling_menuitem(self, item):
            self.snufflings_menu.addAction(item)
            def delete_item():
                self.snufflings_menu.removeAction(item)
            return delete_item
        
        
        def add_traces(self, traces):
            if traces:
                mtf = pyrocko.pile.MemTracesFile(None, traces)
                self.pile.add_file(mtf)
                ticket = (self.pile, mtf)
                return ticket
            else:
                return (None,None)
            
        def release_data(self, tickets):
            for ticket in tickets:
                pile, mtf = ticket
                if pile is not None:
                     pile.remove_file(mtf)
                   
        def periodical(self):
            if self.menuitem_watch.isChecked():
                if self.pile.reload_modified():
                    self.update()
    
        def get_pile(self):
            return self.pile
        
        def pile_changed(self, what):
            self.pile_has_changed = True
           
        def set_gathering(self, gather=None, order=None, color=None):
            
            if gather is None:
                gather = lambda tr: tr.nslc_id
                
            if order is None:
                order = lambda a,b: cmp(a, b)
            
            if color is None:
                color = lambda tr: tr.location
            
            self.gather = gather
            keys = self.pile.gather_keys(gather, self.trace_filter) 
            self.color_gather = color
            self.color_keys = self.pile.gather_keys(color)
            previous_ntracks = self.ntracks
            self.set_ntracks(len(keys))
            if self.shown_tracks_range is None or previous_ntracks == 0 or previous_ntracks != self.ntracks:
                l, h = 0, min(self.ntracks_shown_max, self.ntracks)
            else:
                l, h = self.shown_tracks_range
            
            
            self.set_tracks_range((l,h))

            self.track_keys = sorted(keys, cmp=order)
            self.key_to_row = dict([ (key, i) for (i,key) in enumerate(self.track_keys) ])
            
            inrange = lambda x,r: r[0] <= x and x < r[1]
            
            def trace_selector(trace):
                gt = self.gather(trace)
                return (gt in self.key_to_row and
                   inrange(self.key_to_row[gt], self.shown_tracks_range))
        
            if self.trace_filter is not None:
                self.trace_selector = lambda x: self.trace_filter(x) and trace_selector(x)
            else:
                self.trace_selector = trace_selector

            if self.tmin == working_system_time_range[0] and self.tmax == working_system_time_range[1]:
                self.set_time_range(self.pile.get_tmin(), self.pile.get_tmax())
        
        def set_time_range(self, tmin, tmax):
            if tmin is None:
                tmin = working_system_time_range[0]

            if tmax is None:
                tmax = working_system_time_range[1]

            if tmin > tmax:
                tmin, tmax = tmax, tmin
                
            if tmin == tmax:
                tmin -= 1.
                tmax += 1.
            
            tmin = max(working_system_time_range[0], tmin)
            tmax = min(working_system_time_range[1], tmax)
                    
            if (tmax - tmin < self.min_deltat):
                m = (tmin + tmax) / 2.
                tmin = m - self.min_deltat/2.
                tmax = m + self.min_deltat/2.
                
            self.time_projection.set_in_range(tmin,tmax)
            self.tmin, self.tmax = tmin, tmax
        
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
            if fn:
                f = open(fn,'w')
                for marker in self.markers:
                    f.write("%s\n" % marker)
                f.close()
                
            
        def read_picks(self):
            fn = QFileDialog.getOpenFileName(self,)
            if fn:
                self.markers.extend(Marker.load_markers(fn))
            
        def add_marker(self, marker):
            self.markers.append(marker)
        
        def add_markers(self, markers):
            self.markers.extend(markers)
        
        def remove_marker(self, marker):
            self.markers.remove(marker)
        
        def remove_markers(self, markers):
            for marker in markers:
                if marker in self.markers:
                    self.markers.remove(marker)
        
        def set_markers(self, markers):
            self.markers = markers
    
        def selected_markers(self):
            return [ marker for marker in self.markers if marker.is_selected() ]
    
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
                self.interactive_range_change_time = time.time()
                
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
                    
                    marker_nslc_ids = marker.get_nslc_ids()
                    if not marker_nslc_ids:
                        return marker
                    
                    for nslc_id in marker_nslc_ids:
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
                    
                    marker_nslc_ids = marker.get_nslc_ids()
                    if not marker_nslc_ids:
                        xstate = True
                    
                    for nslc_id in marker_nslc_ids:
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
                
            elif key_event.text() == ':':
                self.emit(SIGNAL('want_input()'))

            elif key_event.text() == 'f':
                if self.window().windowState() & Qt.WindowFullScreen:
                    self.window().showNormal()
                else:
                    self.window().showFullScreen()
                
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

        def set_ntracks(self, ntracks):
            if self.ntracks != ntracks:
                self.ntracks = ntracks
                if self.shown_tracks_range is not None:
                    l,h = self.shown_tracks_range
                else:
                    l,h = 0,self.ntracks

                self.emit(SIGNAL('tracks_range_changed(int,int,int)'), self.ntracks, l,h) 
        
        def set_tracks_range(self, range, start=None):
            
            l,h = range
            l = min(self.ntracks-1, l)
            h = min(self.ntracks, h)
            l = max(0,l)
            h = max(1,h)
            
            if start is None:
                start = float(l)
            
            if self.shown_tracks_range != (l,h):
                self.shown_tracks_range = l,h
                self.shown_tracks_start = start

                self.emit( SIGNAL('tracks_range_changed(int,int,int)'), self.ntracks, l,h)

        def scroll_tracks(self, shift):
            shown = self.shown_tracks_range
            shiftmin = -shown[0]
            shiftmax = self.ntracks-shown[1]
            shift = max(shiftmin, shift)
            shift = min(shiftmax, shift)
            shown = shown[0] + shift, shown[1] + shift
            
            self.set_tracks_range((int(shown[0]), int(shown[1])))

            self.update()
            
        def zoom_tracks(self, anchor, delta):
            ntracks_shown = self.shown_tracks_range[1]-self.shown_tracks_range[0]
            
            if (ntracks_shown == 1 and delta <= 0) or (ntracks_shown == self.ntracks and delta >= 0):
                return
            
            ntracks_shown += int(round(delta))
            ntracks_shown = min(max(1, ntracks_shown), self.ntracks)
            
            u = self.shown_tracks_start
            nu = max(0., u-anchor*delta)
            nv = nu + ntracks_shown
            if nv > self.ntracks:
                nu -= nv - self.ntracks
                nv -= nv - self.ntracks
            
            self.set_tracks_range((int(round(nu)), int(round(nv))), nu)
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
            
            logger.debug('Time spent drawing: %.3f %.3f %.3f %.3f %.3f' % (self.timer_draw - self.timer_cutout))
            logger.debug('Time spent processing: %.3f %.3f %.3f %.3f %.3f' % self.timer_cutout.get())
            
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
            if self.trace_filter is not None:
                tselector = lambda x: selector(x) and self.trace_filter(x)
            else:
                tselector = selector

            traces = list(self.pile.iter_traces(group_selector=selector, trace_selector=tselector))
            traces.sort( lambda a,b: cmp(a.full_id, b.full_id))

            def drawbox(itrack, istyle, traces):
                v_projection = track_projections[itrack]
                dvmin = v_projection(0.)
                dvmax = v_projection(1.)
                dtmin = time_projection.clipped(traces[0].tmin)
                dtmax = time_projection.clipped(traces[-1].tmax)
                
                style = box_styles[istyle%len(box_styles)]
                rect = QRectF( dtmin, dvmin, float(dtmax-dtmin), dvmax-dvmin )
                p.fillRect(rect, style.fill_brush)
                p.setPen(style.frame_pen)
                p.drawRect(rect)

            
            traces_by_style = {}
            for itr, tr in enumerate(traces):
                gt = self.gather(tr)
                if gt not in self.key_to_row:
                    continue

                itrack = self.key_to_row[gt]
                if not itrack in track_projections: continue
               
                istyle = self.trace_styles.get((tr.full_id, tr.deltat), 0)
                
                
                if len(traces) < 1000:
                    drawbox(itrack, istyle, [tr])
                else:
                    if (itrack, istyle) not in traces_by_style:
                        traces_by_style[itrack, istyle] = []
                    traces_by_style[itrack, istyle].append(tr)

            for (itrack, istyle), traces in traces_by_style.iteritems():
                drawbox(itrack, istyle, traces) 
        
        def drawit(self, p, printmode=False, w=None, h=None):
            """This performs the actual drawing."""
            
            self.timer_draw.start()
    
            if self.gather is None:
                self.set_gathering()
    
            if self.pile_has_changed:
                if not self.sortingmode_change_delayed():
                    self.sortingmode_change()
                    
                    if self.menuitem_showboxes.isChecked():
                        self.determine_box_styles()
                
                    self.pile_has_changed = False

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
                    if not self.menuitem_fixscalerange.isChecked():
                        self.old_data_ranges = data_ranges
                    else:
                        data_ranges.update(self.old_data_ranges)
                        
                    for trace in processed_traces:
                        
                        gt = self.gather(trace)
                        if gt not in self.key_to_row:
                            continue
                        
                        itrack = self.key_to_row[gt]
                        if itrack in track_projections:
                            if itrack not in self.track_to_nslc_ids:
                                self.track_to_nslc_ids[itrack] = set()
                            self.track_to_nslc_ids[itrack].add( trace.nslc_id )
                            
                            track_projection = track_projections[itrack]
                            data_range = data_ranges[self.scaling_key(trace)]
                            ymin, ymax, yinc = yscaler.make_scale( data_range )
                            track_projection.set_in_range(ymax,ymin)
        
                            vdata = track_projection( self.gain*trace.get_ydata() )
                            
                            udata_min = float(self.time_projection(trace.tmin))
                            udata_max = float(self.time_projection(trace.tmin+trace.deltat*(vdata.size-1)))
                            udata = num.linspace(udata_min, udata_max, vdata.size)
                            
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
                        plabel = ' '.join([ str(x) for x in key if x is not None ])
                        lx = 10
                        ly = self.track_to_screen(itrack+0.5)
                        draw_label( p, lx, ly, plabel, label_bg, 'BL')
                        
                        if (self.menuitem_showscalerange.isChecked() and itrack in min_max_for_annot):
                            if min_max_for_annot[itrack] is not None:
                                plabel = '(%.2g, %.2g)' % min_max_for_annot[itrack]
                            else:
                                plabel = 'Mixed Scales!'
                            label = QString( plabel)
                            
                            lx = w-10
                            draw_label( p, lx, ly, plabel, label_bg, 'BR')
            
            self.timer_draw.stop()
                            
        def prepare_cutout(self, tmin, tmax, trace_selector=None, degap=True):
            
            self.timer_cutout.start()
            
            tmin_ = tmin
            tmax_ = tmax
            
            # determine padding and downampling requirements
            if self.lowpass is not None:
                deltat_target = 1./self.lowpass * 0.25
                ndecimate = min(50, max(1, int(round(deltat_target / self.min_deltat))))
                tpad = 1./self.lowpass * 2.
            else:
                ndecimate = 1
                tpad = self.min_deltat*5.
                
            if self.highpass is not None:
                tpad = max(1./self.highpass * 2., tpad)
            
            # should we show the traces?
            nsee_points_per_trace = 5000*10
            see_data_range = ndecimate*nsee_points_per_trace*self.min_deltat
            show_traces = (tmax_ - tmin_) < see_data_range
            
            # fetch more than needed?
            if self.menuitem_liberal_fetch.isChecked():
                tlen = pyrocko.trace.nextpow2((tmax-tmin)*1.5)
                tmin = math.floor(tmin/tlen) * tlen
                tmax = math.ceil(tmax/tlen) * tlen
                     
            fft_filtering = self.menuitem_fft_filtering.isChecked()
            lphp = self.menuitem_lphp.isChecked()
            ads = self.menuitem_allowdownsampling.isChecked()
            
            # state vector to decide if cached traces can be used
            vec = (tmin, tmax, trace_selector, degap, self.lowpass, self.highpass, fft_filtering, lphp,
                show_traces, self.rotate, self.shown_tracks_range,
                ads, self.pile.get_update_count())
                
            if (self.old_vec and 
                self.old_vec[0] <= vec[0] and vec[1] <= self.old_vec[1] and
                vec[2:] == self.old_vec[2:] and not (self.reloaded or self.menuitem_watch.isChecked())):
                
                logger.debug('Using cached traces')
                processed_traces = self.old_processed_traces
                
            else:
                self.old_vec = vec
                
                tpad = min(tmax-tmin, tpad)
                tpad = max(self.min_deltat*5., tpad)
                    
                processed_traces = []
                
                if show_traces:
                    
                    for traces in self.pile.chopper( tmin=tmin, tmax=tmax, tpad=tpad,
                                                    want_incomplete=True,
                                                    degap=degap,
                                                    keep_current_files_open=True, 
                                                    trace_selector=trace_selector,
                                                    accessor_id=id(self)):
                        for trace in traces:
                            
                            if not (trace.meta and 'tabu' in trace.meta and trace.meta['tabu']):
                            
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
            
            chopped_traces = []
            for trace in processed_traces:
                try:
                    ctrace = trace.chop(tmin_-trace.deltat*4.,tmax_+trace.deltat*4., inplace=False)
                except pyrocko.trace.NoData:
                    continue
                    
                if len(ctrace.get_ydata()) < 2: continue
                
                chopped_traces.append(ctrace)
            
            self.timer_cutout.stop()
            return chopped_traces
        
        def scalingbase_change(self, ignore):
            for menuitem, scalingbase in self.menuitems_scalingbase:
                if menuitem.isChecked():
                    self.scalingbase = scalingbase
        
        def scalingmode_change(self, ignore):
            for menuitem, scaling_key in self.menuitems_scaling:
                if menuitem.isChecked():
                    self.scaling_key = scaling_key
    
        def s_sortingmode_change(self, ignore=None):
            for menuitem, valfunc in self.menuitems_ssorting:
                if menuitem.isChecked():
                    self._ssort = valfunc
            
            self.sortingmode_change()
    
        def sortingmode_change(self, ignore=None):
            for menuitem, (gather, order, color) in self.menuitems_sorting:
                if menuitem.isChecked():
                    self.set_gathering(gather, order, color)
                    
            self.sortingmode_change_time = time.time()
            
        def lowpass_change(self, value, ignore=None):
            if num.isfinite(value):
                self.lowpass = value
            else:
                self.lowpass = None
            self.passband_check()
            self.update()
            
        def highpass_change(self, value, ignore=None):
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
            
        def set_sortingmode_change_delay_time(self, dt):
            self.sortingmode_change_delay_time = dt
            
        def sortingmode_change_delayed(self):
            now = time.time()
            return (self.sortingmode_change_delay_time is not None and 
                now - self.sortingmode_change_time < self.sortingmode_change_delay_time)
            
        def following(self):
            return self.follow_timer is not None and not self.following_interrupted()
        
        def following_interrupted(self, now=None):
            if now is None:
                now = time.time()
            return now - self.interactive_range_change_time < self.interactive_range_change_delay_time

        def follow(self, tlen, interval=50):
            self.follow_time = tlen
            self.follow_timer = QTimer(self)
            self.connect( self.follow_timer, SIGNAL("timeout()"), self.follow_update ) 
            self.follow_timer.setInterval(interval)
            self.follow_timer.start()
            
        def follow_update(self):
            now = time.time()
            if self.following_interrupted(now):
                return
            self.set_time_range(now-self.follow_time, now)
            self.update()
     
        def myclose(self):
            self.window().close()
            
        def set_error_message(self, key, value):
            if value is None:
                if key in self.error_messages:
                    del self.error_messages[key]
            else:
                self.error_messages[key] = value
        
        def inputline_changed(self, text):
            line = str(text)
            toks = line.split()
            
            if len(toks) in (1,2):
                command= toks[0].lower()
                x = { 'n': '%s*.*.*.*', 's': '*.%s*.*.*', 'l': '*.*.%s*.*', 'c': '*.*.*.%s*' }
                if command in x:
                    if len(toks) == 2:
                        pattern = x[toks[0]] % toks[1].rstrip('*')
                        self.set_quick_filter_pattern(pattern, line)
                    else:
                        self.set_quick_filter_pattern(None)
                    
                    self.update()
                
        def inputline_finished(self, text):
            toks = text.split()
            clearit, hideit, error = False, True, None
            if len(toks) >= 1:
                command = toks[0].lower()
                try:
                    if command in ('hide', 'unhide'):
                        if len(toks) in (2,3):
                            if len(toks) == 2:
                                pattern = toks[1]
                            elif len(toks) == 3:
                                x = { 'n': '%s.*.*.*', 's': '*.%s.*.*', 'l': '*.*.%s.*', 'c': '*.*.*.%s' }
                                if toks[1] in x:
                                    pattern = x[toks[1]] % toks[2]
                            
                            if command == 'hide':
                                self.add_blacklist_pattern( pattern )
                            else:
                                self.remove_blacklist_pattern( pattern )
                        
                        elif command == 'unhide' and len(toks) == 1:
                            self.clear_blacklist()
                        
                        clearit = True
                        
                        self.update()
                        
                    elif command in ('n', 's', 'l', 'c'):
                        pass
                    
                    else:
                        raise PileOverviewException('No such command: %s' % command)
                        
                except PileOverviewException, e:
                    error = str(e)
                    hideit = False
                
            return clearit, hideit, error
        
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
        

class PileViewer(QFrame):
    '''PileOverview + Controls'''
    
    def __init__(self, pile, ntracks_shown_max=20, use_opengl=False, add_panel_hook=None, *args):
        apply(QFrame.__init__, (self,) + args)
        
        if use_opengl:
            self.pile_overview = GLPileOverview(pile, ntracks_shown_max=ntracks_shown_max, add_panel_hook=add_panel_hook)
        else:
            self.pile_overview = PileOverview(pile, ntracks_shown_max=ntracks_shown_max, add_panel_hook=add_panel_hook)
        
        layout = QGridLayout()
        self.setLayout( layout )
        
        self.inputline = QLineEdit()
        self.connect(self.inputline, SIGNAL('returnPressed()'), self.inputline_returnpressed)
        self.connect(self.inputline, SIGNAL('editingFinished()'), self.inputline_finished)
        self.connect(self.inputline, SIGNAL('textEdited(QString)'), self.inputline_changed)
        self.inputline.setFocusPolicy(Qt.ClickFocus)
        self.inputline.hide()
        
        self.inputline_error_str = None
        
        self.inputline_error = QLabel()
        self.inputline_error.hide()

        layout.addWidget( self.inputline, 0, 0, 1, 2 )
        layout.addWidget( self.inputline_error, 1, 0, 1, 2 )        
        layout.addWidget( self.pile_overview, 2, 0 )
        
        scrollbar = QScrollBar(Qt.Vertical)
        self.scrollbar = scrollbar
        layout.addWidget( scrollbar, 2, 1 )
        self.connect(self.scrollbar, SIGNAL('valueChanged(int)'), self.scrollbar_changed)
        self.block_scrollbar_changes = False
        
        self.connect(self.pile_overview, SIGNAL('want_input()'), self.inputline_show)
        self.connect(self.pile_overview, SIGNAL("tracks_range_changed(int,int,int)"), self.tracks_range_changed)

    def inputline_show(self):
        self.inputline.show()
        self.inputline.setFocus(Qt.OtherFocusReason)
        self.inputline.selectAll()
 
    def inputline_set_error(self, string):
        self.inputline_error_str = string
        self.inputline.setPalette( pyrocko.gui_util.get_err_palette() )
        self.inputline.selectAll()
        self.inputline_error.setText(string)
        self.inputline_error.show()
        
    def inputline_clear_error(self):
        if self.inputline_error_str:
            self.inputline.setPalette( QApplication.palette() )
            self.inputline_error_str = None
            self.inputline_error.clear()
            self.inputline_error.hide()
            
    def inputline_changed(self, line):
        self.pile_overview.inputline_changed(str(line))
        self.inputline_clear_error()
        
    def inputline_returnpressed(self):
        self.inputline_finished(returnpressed=True)
        
    def inputline_finished(self, returnpressed=False):
        if returnpressed:
            line = str(self.inputline.text())
            clearit, hideit, error = self.pile_overview.inputline_finished(line)
        else:
            clearit, hideit, error = False, True, None
            
        if error:
            self.inputline_set_error(error)
        
        if clearit:
            self.inputline.blockSignals(True)
            qpat, qinp = self.pile_overview.get_quick_filter_pattern()
            if qpat is None:
                self.inputline.clear()
            else:
                self.inputline.setText(qinp)
            self.inputline.blockSignals(False)
        
        if hideit:
            self.pile_overview.setFocus(Qt.OtherFocusReason) 
            self.inputline.hide()
            
    def tracks_range_changed(self, ntracks, ilo, ihi):
        if self.block_scrollbar_changes:
            return
                
        self.scrollbar.blockSignals(True)
        self.scrollbar.setPageStep(ihi-ilo)
        vmax = max(0,ntracks-(ihi-ilo))
        self.scrollbar.setRange(0, vmax)
        self.scrollbar.setValue(ilo)
        self.scrollbar.setHidden(vmax == 0)
        self.scrollbar.blockSignals(False)

    def scrollbar_changed(self, value):
        self.block_scrollbar_changes = True
        ilo = value
        ihi = ilo + self.scrollbar.pageStep()
        self.pile_overview.set_tracks_range((ilo, ihi))
        self.block_scrollbar_changes = False
        self.update_contents()
        
    def controls(self):
        frame = QFrame(self)
        layout = QGridLayout()
        frame.setLayout(layout)
        
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
        return frame
    
    def get_view(self):
        return self.pile_overview
    
    def update_contents(self):
        self.pile_overview.update()
    
    def get_pile(self):
        return self.pile_overview.get_pile()

from forked import Forked
class SnufflerOnDemand(QApplication, Forked):
    def __init__(self, *args):
        apply(QApplication.__init__, (self,) + args)
        Forked.__init__(self, flipped=True)
        self.timer = QTimer( self )
        self.connect( self.timer, SIGNAL("timeout()"), self.periodical ) 
        self.timer.setInterval(100)
        self.timer.start()
        self.caller_has_quit = False
        self.viewers = {}
        self.windows = []
        
    def dispatch(self, command, args, kwargs):
        method = getattr(self, command)
        method(*args, **kwargs)
        
    def add_traces(self, traces, viewer_id='default'):
        viewer = self.get_viewer(viewer_id)
        pile = viewer.get_pile()
        memfile = pyrocko.pile.MemTracesFile(None, traces)
        pile.add_file(memfile)
        viewer.update_contents()
        
    def periodical(self):
        if not self.caller_has_quit:
            self.caller_has_quit = not self.process()
            
    def get_viewer(self, viewer_id):
        if viewer_id not in self.viewers:
            self.new_viewer(viewer_id)
            
        return self.viewers[viewer_id]
            
    def new_viewer(self, viewer_id):
        pile = pyrocko.pile.Pile()
        pile_viewer = PileViewer(pile)
        win = QMainWindow()
        win.setCentralWidget(pile_viewer)
        win.setWindowTitle( "Snuffler (%s)" % (viewer_id) )        
        win.show()
        self.viewers[viewer_id] = pile_viewer
        self.windows.append(win)
        
    def run(self):
        self.exec_()
    

def snuffle(traces=None, viewer_id='default'):
    
    if Global.appOnDemand is None:
        app = Global.appOnDemand = SnufflerOnDemand([])
        
    app = Global.appOnDemand
    if traces is not None:
        app.call('add_traces', traces, viewer_id)
    
    
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
