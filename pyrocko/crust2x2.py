'''Interface to use CRUST2.0 model by Laske, Masters and Reif.

All functions defined in this module return SI units (m, m/s, kg/m^3).

.. note:: 
  Please refer to the REM web site if you use this model: 

    http://igppweb.ucsd.edu/~gabi/rem.html

  or

    Bassin, C., Laske, G. and Masters, G., The Current Limits of Resolution for
    Surface Wave Tomography in North America, EOS Trans AGU, 81, F897, 2000. A 
    description of CRUST 5.1 can be found in: Mooney, Laske and Masters, Crust 5.1:
    a global crustal model at 5x5 degrees, JGR, 103, 727-747, 1998.
'''

import numpy as num
import os, sys
import copy
from StringIO import StringIO

LICE, LWATER, LSOFTSED, LHARDSED, LUPPERCRUST, LMIDDLECRUST, LLOWERCRUST, LBELOWCRUST = range(8)

class Crust2Profile:
    '''Representation of a CRUST2.0 key profile.'''
    
    layer_names = ('ice', 'water', 'soft sed.', 'hard sed.', 'upper crust', 'middle crust', 'lower crust')
    
    def __init__(self, ident, name, vp, vs, rho, thickness, elevation):
        self._ident = ident
        self._name = name
        self._vp = vp
        self._vs = vs
        self._rho = rho
        self._thickness = thickness
        self._crustal_thickness = None
        self._elevation = elevation
    
    def get_weeded(self, include_waterlayer=False):
        '''Get layers used in the profile.
        
        :param include_waterlayer: include water layer if `True`. Default is `False`
        
        :returns: NumPy array with rows `depth`, `vp`, `vs`, `density`
        '''
        depth = 0.
        layers = []
        for ilayer, thickness, vp, vs, rho in zip(range(8), 
            self._thickness, self._vp[:-1], self._vs[:-1], self._rho[:-1]):
            if thickness == 0.0:
                continue
            
            if not include_waterlayer and ilayer==LWATER:
                continue
            
            layers.append([depth, vp,vs, rho])
            layers.append([depth+thickness, vp,vs, rho])
            depth += thickness
            
        layers.append([depth, 
            self._vp[LBELOWCRUST],
            self._vs[LBELOWCRUST],
            self._rho[LBELOWCRUST]])
        
        return num.array(layers).T

    def get_layer(self, ilayer):
        '''Get parameters for a layer.
        
        :param ilayer: id of layer
        :returns: thickness, vp, vs, density
        '''
        
        if ilayer == LBELOWCRUST:
            thickness = num.Inf
        else:
            thickness = self._thickness[ilayer]
            
        return thickness, self._vp[ilayer], self._vs[ilayer], self._rho[ilayer]
            
    def set_elevation(self, elevation):
        self._elevation = elevation
        
    def set_layer_thickness(self, ilayer, thickness):
        self._thickness[ilayer] = thickness
        
    def elevation(self):
        return self._elevation
    
    def __str__(self):
        
        vvp, vvs, vrho, vthi = self.averages()
        
        return '''type, name:              %s, %s
elevation:               %15.5g
crustal thickness:       %15.5g
average vp, vs, rho:     %15.5g %15.5g %15.5g
mantle ave. vp, vs, rho: %15.5g %15.5g %15.5g

%s''' % (self._ident, self._name, self._elevation, vthi, vvp, vvs, vrho,
    self._vp[LBELOWCRUST],self._vs[LBELOWCRUST],self._rho[LBELOWCRUST],
    '\n'.join( [ '%15.5g %15.5g %15.5g %15.5g   %s' % x for x in zip(
        self._thickness, self._vp[:-1], self._vs[:-1], self._rho[:-1],
        Crust2Profile.layer_names ) ])
      )
   
    def crustal_thickness(self):
        '''Get total crustal thickness
        
        Takes into account ice layer.
        Does not take into account water layer.
        '''
        
        return num.sum(self._thickness[3:]) + self._thickness[LICE]

    def averages(self):
        '''Get crustal averages for vp, vs and density and total crustal thickness,
      
        Takes into account ice layer.
        Does not take into account water layer.
        '''
        
        vthi = self.crustal_thickness()
        vvp = num.sum(self._thickness[3:] / self._vp[3:-1]) + self._thickness[LICE] / self._vp[LICE]
        vvs = num.sum(self._thickness[3:] / self._vs[3:-1]) + self._thickness[LICE] / self._vs[LICE]
        vrho = num.sum(self._thickness[3:] * self._rho[3:-1]) + self._thickness[LICE] * self._rho[LICE]
            
        vvp = vthi / vvp
        vvs = vthi / vvs
        vrho = vrho / vthi
    
        return vvp, vvs, vrho, vthi
        
def sa2arr(sa):
    return num.array([ float(x) for x in sa ], dtype=num.float)

def wrap(x, mi, ma):
    if mi <= x and x <= ma: return x
    return x - math.floor((x-mi)/(ma-mi)) * (ma-mi)

def clip(x, mi, ma):
    return min(max(mi,x),ma)

class Crust2:
    '''Access CRUST2.0 model.
        
        :param directory: Directory with the data files which contain the 
            CRUST2.0 model data. If this is set to `None`, builtin CRUST2.0 files 
            are used.''' 
            
    fn_keys      = 'CNtype2_key.txt'
    fn_elevation = 'CNelevatio2.txt'
    fn_map       = 'CNtype2.txt'
    
    nlo = 180
    nla = 90
    
    _instance = None

    def __init__(self, directory=None):
    
        self._directory = directory
        self._typemap = None
        self._load_crustal_model()
        
    def get_profile(self, lat, lon):
        '''Get crustal profile at a specific location.
        
        :param lat: latitude
        :param lon: longitude
        :rtype: instance of :py:class:`Crust2Profile`
        '''
        
        return self._typemap[self._indices(float(lat),float(lon))]
        
    def _indices(self, lat,lon):
        lat = clip(lat, -90., 90.)
        lon = wrap(lon, -180., 180.)
        dlo = 360./Crust2.nlo
        dla = 180./Crust2.nla
        cola = 90.-lat
        ilat = clip(int(cola/dla), 0, Crust2.nla-1)
        ilon = int((lon+180.)/dlo)%Crust2.nlo
        return ilat, ilon
        
    def _load_crustal_model(self):

        if self._directory is not None:
            path_keys = os.path.join(self._directory, Crust2.fn_keys)
            f = open(path_keys, 'r')
        else:
            from crust2x2_data import decode, type2_key, type2, elevation
            f = StringIO(decode(type2_key))

        # skip header
        for i in range(5):
            f.readline()
                    
        profiles = {}
        while True:
            line = f.readline()
            if not line:
                break
            ident, name = line.split(None, 1)
            line = f.readline()
            vp = sa2arr(line.split()) * 1000.
            line = f.readline()
            vs = sa2arr(line.split()) * 1000.
            line = f.readline()
            rho = sa2arr(line.split()) * 1000.
            line = f.readline()
            toks = line.split()
            thickness = sa2arr(toks[:-2]) * 1000.
            
            profiles[ident] = Crust2Profile(ident.strip(), name.strip(), vp, vs, rho, thickness, 0.0)
            
        f.close()
        
        if self._directory is not None:
            path_map = os.path.join(self._directory, Crust2.fn_map)
            f = open(path_map, 'r')
        else:
            f = StringIO(decode(type2))

        f.readline() # header
            
        amap = {}
        for ila, line in enumerate(f):
            keys = line.split()[1:]
            for ilo, key in enumerate(keys):
                amap[ila,ilo] = copy.deepcopy(profiles[key])
            
        f.close()
        
        if self._directory is not None:
            path_elevation = os.path.join(self._directory, Crust2.fn_elevation)
            f = open(path_elevation, 'r')
    
        else:
            f = StringIO(decode(elevation))
           
        f.readline()
        for ila, line in enumerate(f):
            for ilo, s in enumerate(line.split()[1:]):
                p = amap[ila,ilo]
                p.set_elevation(float(s))
                if p.elevation() < 0.:
                    p.set_layer_thickness(LWATER, -p.elevation())
        
        f.close()
        
        self._typemap = amap

    @staticmethod
    def instance():
        if Crust2._instance is None:
            Crust2._instance = Crust2()

        return Crust2._instance


def get_profile(lat, lon):
   crust2 = Crust2.instance()
   return crust2.get_profile(lat,lon)

        
def plot_crustal_thickness(crust2=None, filename='crustal_thickness.pdf'):
    '''Create a quick and dirty plot of the crustal thicknesses defined in CRUST2.0.'''

    if crust2 is None:
        crust2 = Crust2.instance()
   
    def func(lat,lon):
        return crust2.get_profile(lat,lon).crustal_thickness(), 
    
    plot(func, filename, zscaled_unit='km', zscaled_unit_factor=0.001)
    
def plot_vp_belowcrust(crust2=None, filename='vp_below_crust.pdf'):
    '''Create a quick and dirty plot of vp below the crust, as defined in CRUST2.0.'''
    if crust2 is None:
        crust2 = Crust2.instance()
        
    def func(lat,lon):
        return crust2.get_profile(lat,lon).get_layer(LBELOWCRUST)[1]
    
    plot(func, filename, zscaled_unit='km/s', zscaled_unit_factor=0.001)
    
    
def plot(func, filename, **kwargs):
    nlats, nlons = 91, 181
    lats = num.linspace(-90., 90., nlats)
    lons = num.linspace(-180.,180., nlons)
        
    vecfunc = num.vectorize( func, [ num.float ] )
    latss, lonss = num.meshgrid(lats, lons)         
    thickness = vecfunc(latss, lonss)
    
    import gmtpy
    cm = gmtpy.cm
    marg = (1.5*cm, 2.5*cm, 1.5*cm, 1.5*cm)
    p = gmtpy.Simple(width=20*cm, height=10*cm, margins=marg, 
        with_palette=True, **kwargs)
    
    p.density_plot(gmtpy.tabledata(lons,lats, thickness.T))
    p.save(filename)

