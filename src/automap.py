#!/usr/bin/env python

import math
import random
import shutil
import os.path as op
import logging

from subprocess import check_call, Popen, PIPE
from cStringIO import StringIO

import numpy as num


from pyrocko.guts import Object, Float, Bool, Int, Tuple
from pyrocko import orthodrome as od
from pyrocko import gmtpy, topo, config

logger = logging.getLogger('pyrocko.automap')

earthradius = 6371000.0
r2d = 180./math.pi
d2r = 1./r2d
km = 1000.
d2m = d2r*earthradius
m2d = 1./d2m
cm = gmtpy.cm


def point_in_region(p, r):
    p = [num.mod(x, 360.) for x in p]
    r = [num.mod(x, 360.) for x in r]
    if r[0] <= r[1]:
        blon = r[0] <= p[0] <= r[1]
    else:
        blon = not (r[1] < p[0] < r[0])
    if r[2] <= r[3]:
        blat = r[2] <= p[1] <= r[3]
    else:
        blat = not (r[3] < p[1] < r[2])

    return blon and blat


def darken(c, f=0.7):
    return (c[0]*f, c[1]*f, c[2]*f)


def corners(lon, lat, w, h):
    ll_lat, ll_lon = od.ne_to_latlon(lat, lon, -0.5*h, -0.5*w)
    ur_lat, ur_lon = od.ne_to_latlon(lat, lon, 0.5*h, 0.5*w)
    return ll_lon, ll_lat, ur_lon, ur_lat


def extent(lon, lat, w, h, n):
    x = num.linspace(-0.5*w, 0.5*w, n)
    y = num.linspace(-0.5*h, 0.5*h, n)
    slats, slons = od.ne_to_latlon(lat, lon, y[0], x)
    nlats, nlons = od.ne_to_latlon(lat, lon, y[-1], x)
    south = slats.min()
    north = nlats.max()

    wlats, wlons = od.ne_to_latlon(lat, lon, y, x[0])
    elats, elons = od.ne_to_latlon(lat, lon, y, x[-1])
    elons = num.where(elons < wlons, elons + 360., elons)

    if elons.max() - elons.min() > 180 or wlons.max() - wlons.min() > 180.:
        west = -180.
        east = 180.
    else:
        west = wlons.min()
        east = elons.max()

    return topo.positive_region((west, east, south, north))


class NoTopo(Exception):
    pass


class Map(Object):
    lat = Float.T()
    lon = Float.T()
    radius = Float.T()
    width = Float.T(default=20.)
    height = Float.T(default=14.)
    illuminate = Bool.T(default=True)
    skip_feature_factor = Float.T(default=0.02)
    show_grid = Bool.T(default=False)
    show_topo = Bool.T(default=True)
    show_center_mark = Bool.T(default=False)
    illuminate_factor_land = Float.T(default=0.5)
    illuminate_factor_ocean = Float.T(default=0.25)
    color_wet = Tuple.T(3, Int.T(), default=(216, 242, 254))
    color_dry = Tuple.T(3, Int.T(), default=(172, 208, 165))
    topo_resolution_min = Float.T(
        default=40.,
        help='minimum resolution of topography [dpi]')
    topo_resolution_max = Float.T(
        default=200.,
        help='maximum resolution of topography [dpi]')

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._gmt = None
        self._scaler = None
        self._widget = None
        self._corners = None
        self._wesn = None
        self._minarea = None
        self._coastline_resolution = None
        self._rivers = None
        self._dems = None
        self._have_topo_land = None
        self._have_topo_ocean = None
        self._jxyr = None
        self._prep_topo_have = None
        self._labels = []

    def save(self, outpath):
        gmt = self.gmt
        self._draw_labels()
        self._draw_axes()
        gmt = self._gmt
        if outpath.endswith('.eps'):
            tmppath = gmt.tempfilename() + '.eps'
        elif outpath.endswith('.ps'):
            tmppath = gmt.tempfilename() + '.ps'
        else:
            tmppath = gmt.tempfilename() + '.pdf'

        gmt.save(tmppath)

        if any(outpath.endswith(x) for x in ('.eps', '.ps', '.pdf')):
            shutil.copy(tmppath, outpath)
        else:
            convert_graph(tmppath, outpath)

    @property
    def scaler(self):
        if self._scaler is None:
            self._setup_geometry()

        return self._scaler

    @property
    def wesn(self):
        if self._wesn is None:
            self._setup_geometry()

        return self._wesn

    @property
    def widget(self):
        if self._widget is None:
            self.setup()

        return self._widget

    @property
    def jxyr(self):
        if self._jxyr is None:
            self.setup()

        return self._jxyr

    @property
    def gmt(self):
        if self._gmt is None:
            self._setup()

        if self._have_topo_ocean is None:
            self._draw_background()

        return self._gmt

    def _setup(self):
        if not self._widget:
            self._setup_geometry()

        self._setup_lod()
        self._setup_gmt()

    def _setup_geometry(self):
        wpage, hpage = self.width, self.height

        wreg = self.radius * 2.0
        hreg = self.radius * 2.0
        if wpage >= hpage:
            wreg *= wpage/hpage
        else:
            hreg *= hpage/wpage

        self._corners = corners(self.lon, self.lat, wreg, hreg)
        west, east, south, north = extent(self.lon, self.lat, wreg, hreg, 10)

        x, y, z = ((west, east), (south, north), (-6000., 4500.))

        xax = gmtpy.Ax(mode='min-max', approx_ticks=4.)
        yax = gmtpy.Ax(mode='min-max', approx_ticks=4.)
        zax = gmtpy.Ax(mode='min-max', inc=1000., label='Height',
                       scaled_unit='km', scaled_unit_factor=0.001)

        scaler = gmtpy.ScaleGuru(data_tuples=[(x, y, z)], axes=(xax, yax, zax))

        par = scaler.get_params()

        west = par['xmin']
        east = par['xmax']
        south = par['ymin']
        north = par['ymax']

        self._wesn = west, east, south, north
        self._scaler = scaler

    def _setup_lod(self):
        w, e, s, n = self._wesn
        if self.radius > 1500.*km:
            coastline_resolution = 'i'
            rivers = False
        else:
            coastline_resolution = 'f'
            rivers = True

        self._minarea = (self.skip_feature_factor * self.radius/km)**2

        self._coastline_resolution = coastline_resolution
        self._rivers = rivers

        self._prep_topo_have = {}
        self._dems = {}

        cm2inch = gmtpy.cm/gmtpy.inch

        dmin = 2.0 * self.radius * m2d / (self.topo_resolution_max *
                                          (self.height * cm2inch))
        dmax = 2.0 * self.radius * m2d / (self.topo_resolution_min *
                                          (self.height * cm2inch))

        for k in ['ocean', 'land']:
            self._dems[k] = topo.select_dem_names(k, dmin, dmax, self._wesn)
            if self._dems[k]:
                logger.debug('using topography dataset %s for %s'
                             % (','.join(self._dems[k]), k))

    def _setup_gmt(self):
        w, h = self.width, self.height
        scaler = self._scaler

        gmtconf = dict(
            TICK_PEN='1.25p',
            TICK_LENGTH='0.2c',
            ANNOT_FONT_PRIMARY='1',
            ANNOT_FONT_SIZE_PRIMARY='12p',
            LABEL_FONT='1',
            LABEL_FONT_SIZE='12p',
            CHAR_ENCODING='ISOLatin1+',
            BASEMAP_TYPE='fancy',
            PLOT_DEGREE_FORMAT='D',
            PAPER_MEDIA='Custom_%ix%i' % (
                w*gmtpy.cm,
                h*gmtpy.cm),
            GRID_PEN_PRIMARY='thinnest/0/50/0')

        gmt = gmtpy.GMT(config=gmtconf)

        layout = gmt.default_layout()
        mw = 2.0*cm
        layout.set_fixed_margins(mw, mw, 0.5*cm, 0.5*cm)
        widget = layout.get_widget()
        #widget['J'] = ('-JT%g/%g' % (self.lon, self.lat)) + '/%(width)gp'
        #widget['J'] = ('-JA%g/%g' % (self.lon, self.lat)) + '/%(width)gp'
        widget['J'] = ('-JE%g/%g' % (self.lon, self.lat)) + '/%(width)gp'
        #scaler['R'] = '-R%(xmin)g/%(xmax)g/%(ymin)g/%(ymax)g'
        scaler['R'] = '-R%g/%g/%g/%gr' % self._corners

        aspect = gmtpy.aspect_for_projection(*(widget.J() + scaler.R()))
        widget.set_aspect(aspect)

        self._gmt = gmt
        self._layout = layout
        self._widget = widget
        self._jxyr = self._widget.JXY() + self._scaler.R()

    def _draw_background(self):
        self._have_topo_land = False
        self._have_topo_ocean = False
        if self.show_topo:
            self._have_topo = self._draw_topo()

        self._draw_basefeatures()

    def _prep_topo(self, k):
        gmt = self._gmt

        t = None
        demname = None
        for dem in self._dems[k]:
            t = topo.get(dem, self._wesn)
            demname = dem
            if t is not None:
                break

        if not t:
            raise NoTopo()

        if demname not in self._prep_topo_have:
            grdfile = gmt.tempfilename()
            gmtpy.savegrd(
                t.x(), t.y(), t.data, filename=grdfile, naming='lonlat')

            if self.illuminate:
                if k == 'ocean':
                    factor = self.illuminate_factor_ocean
                else:
                    factor = self.illuminate_factor_land

                ilumfn = gmt.tempfilename()
                gmt.grdgradient(
                    grdfile,
                    N='e%g' % factor,
                    A=-45,
                    G=ilumfn,
                    out_discard=True)

                ilumargs = ['-I%s' % ilumfn]
            else:
                ilumargs = []

            self._prep_topo_have[demname] = grdfile, ilumargs

        return self._prep_topo_have[demname]

    def _draw_topo(self):
        widget = self._widget
        scaler = self._scaler
        gmt = self._gmt
        cres = self._coastline_resolution
        minarea = self._minarea

        JXY = widget.JXY()
        R = scaler.R()

        try:
            grdfile, ilumargs = self._prep_topo('ocean')
            gmt.pscoast(D=cres, S='c', A=minarea, *(JXY+R))
            gmt.grdimage(grdfile, C=topo.cpt('light_sea'),
                         *(ilumargs+JXY+R))
            gmt.pscoast(Q=True, *(JXY+R))
            self._have_topo_ocean = True
        except NoTopo:
            self._have_topo_ocean = False

        try:
            grdfile, ilumargs = self._prep_topo('land')
            gmt.pscoast(D=cres, G='c', A=minarea, *(JXY+R))
            gmt.grdimage(grdfile, C=topo.cpt('light_land'),
                         *(ilumargs+JXY+R))
            gmt.pscoast(Q=True, *(JXY+R))
            self._have_topo_land = True
        except NoTopo:
            self._have_topo_land = False

    def _draw_basefeatures(self):
        gmt = self._gmt
        cres = self._coastline_resolution
        rivers = self._rivers
        minarea = self._minarea

        color_wet = self.color_wet
        color_dry = self.color_dry

        if rivers:
            rivers = ['-Ir/0.25p,%s' % gmtpy.color(self.color_wet)]
        else:
            rivers = []

        fill = {}
        if not self._have_topo_land:
            fill['G'] = color_dry

        if not self._have_topo_ocean:
            fill['S'] = color_wet

        gmt.pscoast(
            D=cres,
            W='thinnest,%s' % gmtpy.color(darken(color_dry)),
            A=minarea,
            *(rivers+self._jxyr), **fill)

    def _draw_axes(self):
        gmt = self._gmt
        scaler = self._scaler
        widget = self._widget

        if self.lat > 0.0:
            axes_layout = 'WSen'
        else:
            axes_layout = 'WseN'

        scale_km = gmtpy.nice_value(self.radius/5.) / 1000.

        if self.show_center_mark:
            gmt.psxy(
                in_rows=[[self.lon, self.lat]],
                S='c20p', W='2p,black',
                *self._jxyr)

        if self.show_grid:
            btmpl = ('%(xinc)gg%(xinc)g:%(xlabel)s:/'
                     '%(yinc)gg%(yinc)g:%(ylabel)s:')
        else:
            btmpl = '%(xinc)g:%(xlabel)s:/%(yinc)g:%(ylabel)s:'

        gmt.psbasemap(
            B=(btmpl % scaler.get_params())+axes_layout,
            L=('x%gp/%gp/%g/%g/%gk' % (
                6./7*widget.width(),
                widget.height()/7.,
                self.lon,
                self.lat,
                scale_km)),
            *self._jxyr)

    def _have_coastlines(self):
        gmt = self._gmt
        cres = self._coastline_resolution
        minarea = self._minarea

        checkfile = gmt.tempfilename()

        gmt.pscoast(
            M=True,
            D=cres,
            W='thinnest/black',
            A=minarea,
            out_filename=checkfile,
            *self._jxyr)

        with open(checkfile, 'r') as f:
            for line in f:
                ls = line.strip()
                if ls.startswith('#') or ls.startswith('>') or ls == '':
                    continue
                plon, plat = [float(x) for x in ls.split()]
                if point_in_region((plon, plat), self._wesn):
                    return True

        return False

    def _draw_labels(self):
        if self._labels:
            n = len(self._labels)

            lons, lats, texts, sx, sy, styles = zip(*self._labels)

            sx = num.array(sx, dtype=num.float)
            sy = num.array(sy, dtype=num.float)

            j, _, _, r = self.jxyr
            f = StringIO()
            self.gmt.mapproject(j, r, in_columns=(lons, lats), out_stream=f,
                                D='p')
            f.seek(0)
            data = num.loadtxt(f, ndmin=2)
            xs, ys = data.T
            dxs = num.zeros(n)
            dys = num.zeros(n)

            g = gmtpy.GMT()
            g.psbasemap('-G0', finish=True, *(j, r))
            l, b, r, t = g.bbox()
            h = (t-b)
            w = (r-l)

            for i in xrange(n):
                g = gmtpy.GMT()
                g.pstext(
                    in_rows=[[0, 0, 9, 0, 1, 'BL', texts[i]]],
                    finish=True,
                    R=(0, 1, 0, 1),
                    J='x10p',
                    N=True)

                fn = g.tempfilename()
                g.save(fn)

                (_, stderr) = Popen(
                    ['gs', '-q', '-dNOPAUSE', '-dBATCH', '-r720',
                     '-sDEVICE=bbox', fn],
                    stderr=PIPE).communicate()

                dx, dy = None, None
                for line in stderr.splitlines():
                    if line.startswith('%%HiResBoundingBox:'):
                        l, b, r, t = [float(x) for x in line.split()[-4:]]
                        dx, dy = r-l, t-b

                dxs[i] = dx
                dys[i] = dy

            la = num.logical_and
            anchors_ok = (
                la(xs + sx + dxs < w, ys + sy + dys < h),
                la(xs - sx - dxs > 0., ys - sy - dys > 0.),
                la(xs + sx + dxs < w, ys - sy - dys > 0.),
                la(xs - sx - dxs > 0., ys + sy + dys < h),
            )

            arects = [
                (xs, ys, xs + sx + dxs, ys + sy + dys),
                (xs - sx - dxs, ys - sy - dys, xs, ys),
                (xs, ys - sy - dys, xs + sx + dxs, ys),
                (xs - sx - dxs, ys, xs, ys + sy + dys)]

            def no_points_in_rect(xs, ys, xmin, ymin, xmax, ymax):
                xx = not num.any(la(la(xmin < xs, xs < xmax),
                                    la(ymin < ys, ys < ymax)))
                return xx

            for i in xrange(n):
                for ianch in xrange(4):
                    anchors_ok[ianch][i] &= no_points_in_rect(
                        xs, ys, *[xxx[i] for xxx in arects[ianch]])

            anchor_choices = []
            anchor_take = []
            for i in xrange(n):
                choices = [ianch for ianch in xrange(4)
                           if anchors_ok[ianch][i]]
                anchor_choices.append(choices)
                if choices:
                    anchor_take.append(choices[0])
                else:
                    anchor_take.append(None)

            def roverlaps(a, b):
                return (a[0] < b[2] and b[0] < a[2] and
                        a[1] < b[3] and b[1] < a[3])

            def cost(anchor_take):
                noverlaps = 0
                for i in xrange(n):
                    for j in xrange(n):
                        if i != j:
                            i_take = anchor_take[i]
                            j_take = anchor_take[j]
                            if i_take is None or j_take is None:
                                continue
                            r_i = [xxx[i] for xxx in arects[i_take]]
                            r_j = [xxx[j] for xxx in arects[j_take]]
                            if roverlaps(r_i, r_j):
                                noverlaps += 1

                return noverlaps

            cur_cost = cost(anchor_take)
            imax = 30
            while cur_cost != 0 and imax > 0:
                for i in xrange(n):
                    for t in anchor_choices[i]:
                        anchor_take_new = list(anchor_take)
                        anchor_take_new[i] = t
                        new_cost = cost(anchor_take_new)
                        if new_cost < cur_cost:
                            anchor_take = anchor_take_new
                            cur_cost = new_cost

                imax -= 1

            while cur_cost != 0:
                for i in xrange(n):
                    anchor_take_new = list(anchor_take)
                    anchor_take_new[i] = None
                    new_cost = cost(anchor_take_new)
                    if new_cost < cur_cost:
                        anchor_take = anchor_take_new
                        cur_cost = new_cost
                        break

            anchor_strs = ['BL', 'TR', 'TL', 'BR']
            for i in xrange(n):
                ianchor = anchor_take[i]
                if ianchor is not None:
                    anchor = anchor_strs[ianchor]
                    yoff = [-sy[i], sy[i]][anchor[0] == 'B']
                    xoff = [-sx[i], sx[i]][anchor[1] == 'L']
                    row = (lons[i], lats[i], 9, 0, 1, anchor, texts[i])
                    self.gmt.pstext(
                        in_rows=[row], D='%gp/%gp' % (xoff, yoff), *self.jxyr)

    def add_label(self, lat, lon, text, offset_x=5., offset_y=5.):
        self._labels.append((lon, lat, text, offset_x, offset_y, None))

    def draw_cities(self, nmax_soft=10):
        from pyrocko import geonames
        cities = sorted(list(geonames.load2(
            op.join(config.config().geonames_dir, 'cities1000.txt'),
            region=self.wesn, minpop=0)),
            key=lambda x: x.population)

        minpop = 10**3
        for minpop_new in [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]:
            cities_new = [c for c in cities if c.population > minpop_new and
                          c.name != 'City of London']

            if len(cities_new) == 0 or (
                    len(cities_new) < 3 and len(cities) < nmax_soft*2):
                break

            cities = cities_new
            minpop = minpop_new
            if len(cities) <= nmax_soft:
                break

        lats = [c.latitude for c in cities]
        lons = [c.longitude for c in cities]

        self.gmt.psxy(
            in_columns=(lons, lats),
            S='s5p',
            G='black',
            *self.jxyr)

        for c in cities:
            try:
                text = c.name.encode('iso-8859-1')
            except UnicodeEncodeError:
                text = c.asciiname

            self.add_label(c.latitude, c.longitude, text)

        self._cities_minpop = minpop


def convert_graph(in_filename, out_filename):
    tmp_filename = in_filename + '.oversized'
    check_call(['pdftocairo', '-r', '150', '-png', in_filename, tmp_filename])
    check_call(['convert', tmp_filename+'-1.png',
                '-resize', '50%', out_filename])


def rand(mi, ma):
    mi = float(mi)
    ma = float(ma)
    return random.random() * (ma-mi) + mi


def split_region(region):
    west, east, south, north = topo.positive_region(region)
    if east > 180:
        return [(west, 180., south, north),
                (-180., east-360., south, north)]
    else:
        return [region]


if __name__ == '__main__':
    from pyrocko import util
    util.setup_logging('pyrocko.automap', 'info')

    m = Map(
        lat=rand(40, 70.),
        lon=rand(0., 20.),
        radius=math.exp(rand(math.log(10*km), math.log(2000*km))),
        width=rand(10, 20), height=rand(10, 20),
        show_grid=True,
        show_topo=True,
        illuminate=True)

    m.draw_cities()
    print m
    m.save('map.pdf')
