from pyrocko import orthodrome, config, util

import unittest
import numpy as num
import math, random
import pyrocko.config
config = pyrocko.config.config()

r2d = 180./math.pi
d2r = 1./r2d
km = 1000.

class Loc:
    pass
    
class OrthodromeTestCase(unittest.TestCase):
    
    def testGridDistances(self):
        for i in range(100):
            w,h = 20,15
        
            km = 1000.
            gsize = random.uniform(0.,1.)*2.*10.**random.uniform(4.,7.)
            north_grid, east_grid = num.meshgrid(num.linspace(-gsize/2.,gsize/2.,11) ,
                                                 num.linspace(-gsize/2.,gsize/2.,11) )
            
            north_grid = north_grid.flatten()
            east_grid = east_grid.flatten()
            
            lat_delta = gsize/config.earthradius*r2d*2.
            lon = random.uniform(-180.,180.)
            lat = random.uniform(-90.,90.)
                    
            lat_grid, lon_grid = orthodrome.ne_to_latlon(lat, lon, north_grid, east_grid)
            lat_grid_alt, lon_grid_alt = orthodrome.ne_to_latlon_alternative_method(lat, lon, north_grid, east_grid)
            
            for la, lo, no, ea in zip(lat_grid, lon_grid, north_grid, east_grid):
                a = Loc()
                a.lat = la 
                a.lon = lo
                b = Loc()
                b.lat = lat
                b.lon = lon
                
                cd = orthodrome.cosdelta(a,b)
                assert cd <= 1.0
                d = num.arccos(cd)*config.earthradius
                d2 = math.sqrt(no**2+ea**2)
                assert not (abs(d-d2) > 1.0e-3 and d2 > 1.)

    def test_local_distances(self):
        for reflat, reflon in [
                (0.0, 0.0),
                (10.0, 10.0),
                (90.0, 0.0),
                (-90.0, 0.0),
                (0.0, 180.0),
                (0.0, -180.0),
                (90.0, 180.0) ]:

            north, east = serialgrid(num.linspace(-10*km, 10*km, 21),
                                     num.linspace(-10*km, 10*km, 21))

            lat, lon = orthodrome.ne_to_latlon2(reflat, reflon, north, east)
            north2, east2 = orthodrome.latlon_to_ne2(reflat, reflon, lat, lon)
            dist1 = num.sqrt(north**2 + east**2)
            dist2 = num.sqrt(north2**2 + east2**2)
            dist3 = orthodrome.distance_accurate15nm(reflat, reflon, lat, lon)
            assert num.all(num.abs(dist1-dist2) < 0.0001)
            assert num.all(num.abs(dist1-dist3) < 0.0001)

def serialgrid(x,y):
    return num.repeat(x, y.size), num.tile(y, x.size)

def plot_erroneous_ne_to_latlon():
    import sys
    import gmtpy
    import random
    import subprocess
    import time
    
    while True:
        w,h = 20,15
    
        km = 1000.
        gsize = random.uniform(0.,1.)*4.*10.**random.uniform(4.,7.)
        north_grid, east_grid = num.meshgrid(num.linspace(-gsize/2.,gsize/2.,11) ,
                                             num.linspace(-gsize/2.,gsize/2.,11) )
        
        north_grid = north_grid.flatten()
        east_grid = east_grid.flatten()
        
        lat_delta = gsize/config.earthradius*r2d*2.
        lon = random.uniform(-180.,180.)
        lat = random.uniform(-90.,90.)
        
        print gsize/1000.
        
        lat_grid, lon_grid = ne_to_latlon(lat, lon, north_grid, east_grid)
        lat_grid_alt, lon_grid_alt = ne_to_latlon_alternative_method(lat, lon, north_grid, east_grid)
    
    
        maxerrlat = num.max(num.abs(lat_grid-lat_grid_alt))
        maxerrlon = num.max(num.abs(lon_grid-lon_grid_alt))
        eps = 1.0e-8
        if maxerrlon > eps or maxerrlat > eps:
            print lat, lon, maxerrlat, maxerrlon
        
            gmt = gmtpy.GMT( config={ 'PLOT_DEGREE_FORMAT':'ddd.xxxF',
                                    'PAPER_MEDIA':'Custom_%ix%i' % (w*gmtpy.cm,h*gmtpy.cm),
                                    'GRID_PEN_PRIMARY': 'thinnest/0/50/0' } )
        
            south = max(-85., lat - 0.5*lat_delta)
            north = min(85., lat + 0.5*lat_delta)
                
            lon_delta = lat_delta/math.cos(lat*d2r)
            
            delta = lat_delta/360.*config.earthradius*2.*math.pi
            scale_km = gmtpy.nice_value(delta/10.)/1000.
            
            west = lon - 0.5*lon_delta
            east = lon + 0.5*lon_delta
            
            x,y = (west, east), (south,north)
            xax = gmtpy.Ax(mode='min-max', approx_ticks=4.)
            yax = gmtpy.Ax(mode='min-max', approx_ticks=4.)
            scaler = gmtpy.ScaleGuru( data_tuples=[(x,y)], axes=(xax,yax))    
            scaler['R'] = '-Rg'
            layout = gmt.default_layout()
            mw = 2.5*gmtpy.cm
            layout.set_fixed_margins(mw,mw,mw/gmtpy.golden_ratio,mw/gmtpy.golden_ratio)
            widget = layout.get_widget()
            # widget['J'] =  ('-JT%g/%g'  % (lon, lat)) + '/%(width)gp'
            widget['J'] =  ('-JE%g/%g/%g'  % (lon, lat, min(lat_delta/2.,180.))) + '/%(width)gp'
            aspect = gmtpy.aspect_for_projection( *(widget.J() + scaler.R()) )
            widget.set_aspect(aspect)
            
            if lat > 0:
                axes_layout = 'WSen'
            else:
                axes_layout = 'WseN'
        
            gmt.psbasemap( #B=('%(xinc)gg%(xinc)g:%(xlabel)s:/%(yinc)gg%(yinc)g:%(ylabel)s:' % scaler.get_params())+axes_layout,
                           B='5g5',
                        L=('x%gp/%gp/%g/%g/%gk' % (widget.width()/2., widget.height()/7.,lon,lat,scale_km) ),
                        *(widget.JXY()+scaler.R()) )
            
            gmt.psxy( in_columns=(lon_grid,lat_grid), S='x10p', W='1p/200/0/0', *(widget.JXY()+scaler.R()) )
            gmt.psxy( in_columns=(lon_grid_alt,lat_grid_alt), S='c10p', W='1p/0/0/200', *(widget.JXY()+scaler.R()) )
            
            gmt.save('orthodrome.pdf')
            subprocess.call( [ 'xpdf', '-remote', 'ortho', '-reload' ] )
            time.sleep(2)
        else:
            print 'ok', gsize, lat, lon
            
if __name__ == "__main__":
    util.setup_logging('test_orthodrome', 'warning')
    unittest.main()

            
