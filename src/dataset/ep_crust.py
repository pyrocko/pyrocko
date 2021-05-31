# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
'''
Interface to use EPcrust 0.5x0.5 model by Molinari, Morelli.

Created on Wed May 26 09:10:31 2021
Modified after crust2x2.py
@author: Marin GOvorcin

All functions defined in this module return SI units (m, m/s, kg/m^3).

.. note::
  Please refer to the INGV web site if you use this model:

    http://eurorem.bo.ingv.it/EPcrust_solar/

  or

    "Molinari, I., Morelli, A., 2011. EPcrust: A reference crustal model for
    the european plate. Geophys. J. Int. in press,
    doi: 10.1111/j.1365-246X.2011.04940.x."


The model covers the following region with resolution of 0.5 x 0.5 degrees:

Extended Region:
Latitude:   9 N - 89.5 N           ( original: 20 N - 89.5 N )
Longitude: 56 W - 70 E             (          40 W - 70 E    )

Data is stored in: EPcrust_0_5.txt, where columns are:
    LON   LAT   Topography   sediment_thickness   thickness_upper_crust
    thickness_lower_crust   VP_sediment   VP_upper_cruet   VP_lower_crust
    VS_sediment   VS_upper_crust   VS_lower_crust   RHO_sediment
    RHO_upper_crust RHO_lower_crust

EPcrust is in 3 layer:
- sediment
- upper crust
- lower crust

EPcrust has an additional file with ice_thickness: Ice_thickness_0_5.txt
-----

::

    >>> from pyrocko.dataset import ep_crust
    >>> p = ep_crust.get_profile(10., 20.)
    >>> print(p)
    lat, lon:              10.0, 20.0
    elevation:                           402
    crustal thickness:                 38923
    average vp, vs, rho:              6511.9          3778.7          2840.3

                  0            3810            1940             920   ice
                  0            1500               0            1020   water
                750            2500             984            2093   sediment
              13422            6200            3640            2761   upper crust
              25501            6689            3856            2882   lower crust
    >>> print(p.get_weeded())
    [[    0.   750.   750. 14172. 14172. 39673.]
     [ 2500.  2500.  6200.  6200.  6689.  6689.]
     [  984.   984.  3640.  3640.  3856.  3856.]
     [ 2093.  2093.  2761.  2761.  2882.  2882.]]

===============================================================================
NOTE:
    EPcrust model has 3 layers and ice_thickness. To make this module similar
    to crust2x2, two more layers were added with default vp, vs, rho values
    (from crust2x2) for ice: 3810 1940 920 water: 1500 0 1020


Constants
---------

============== ==============
Layer id       Layer name
============== ==============
LICE           ice
LWATER         water
LSED           sediments
LUPPERCRUST    upper crust
LLOWERCRUST    lower crust
============== ==============

Contents
--------
'''  # noqa

import os
import math

import numpy as num

from pyrocko import util

LICE, LWATER, LSED, LUPPERCRUST, LLOWERCRUST = list(range(5))
# EPcrust provides also ICE thickness which is not includedat the moment


class EPcrust2Profile(object):
    '''Representation of a EPcrust key profile.'''

    layer_names = (
        'ice', 'water', 'sediment', 'upper crust',
        'lower crust')

    def __init__(self, ident, lat, lon, vp, vs, rho, thickness, elevation):
        self._ident = ident
        self._lat = lat
        self._lon = lon
        self._vp = vp
        self._vs = vs
        self._rho = rho
        self._thickness = thickness
        self._crustal_thickness = None
        self._elevation = elevation

    def get_weeded(self):
        '''Get layers used in the profile.

        :returns: NumPy array with rows ``depth``, ``vp``, ``vs``, ``density``
        '''

        depth = 0
        layers = []
        for ilayer, thickness, vp, vs, rho in zip(
                range(5),
                self._thickness,
                self._vp,
                self._vs,
                self._rho):

            if thickness == 0.0:
                continue

            layers.append([depth, vp, vs, rho])
            layers.append([depth+thickness, vp, vs, rho])
            depth += thickness

        return num.array(layers).T

    def get_layer(self, ilayer):
        '''Get parameters for a layer.

        :param ilayer: id of layer
        :returns: thickness, vp, vs, density
        '''

        thickness = self._thickness[ilayer]

        return thickness, self._vp[ilayer], self._vs[ilayer], self._rho[ilayer]

    def set_elevation(self, elevation):
        self._elevation = elevation

    def set_layer_thickness(self, ilayer, thickness):
        self._thickness[ilayer] = thickness

    def __str__(self):

        vvp, vvs, vrho, vthi = self.averages()

        return '''lat, lon:              %s, %s
elevation:               %15.5g
crustal thickness:       %15.5g
average vp, vs, rho:     %15.5g %15.5g %15.5g

%s''' % (self._lat, self._lon, self._elevation, vthi, vvp, vvs, vrho,
         '\n'.join([
             '%15.5g %15.5g %15.5g %15.5g   %s' % x
             for x in zip(
                 self._thickness, self._vp, self._vs, self._rho,
                 EPcrust2Profile.layer_names)]))

    def crustal_thickness(self):
        '''Get total crustal thickness

        Takes into account ice layer.
        Does not take into account water layer.
        '''

        return num.sum(self._thickness[3:]) + self._thickness[LICE]

    def averages(self):
        '''
        Get crustal averages for vp, vs and density and total crustal
        thickness.

        Takes into account ice layer.
        Does not take into account water layer.
        '''

        vthi = self.crustal_thickness()
        vvp = num.sum(self._thickness[3:] / self._vp[3:]) + \
            self._thickness[LICE] / self._vp[LICE]
        vvs = num.sum(self._thickness[3:] / self._vs[3:]) + \
            self._thickness[LICE] / self._vs[LICE]
        vrho = num.sum(self._thickness[3:] * self._rho[3:]) + \
            self._thickness[LICE] * self._rho[LICE]

        vvp = vthi / vvp
        vvs = vthi / vvs
        vrho = vrho / vthi

        return vvp, vvs, vrho, vthi


def _sa2arr(sa):
    return num.array([float(x) for x in sa], dtype=num.float)


def _wrap(x, mi, ma):
    if mi <= x and x <= ma:
        return x

    return x - math.floor((x-mi)/(ma-mi)) * (ma-mi)


def _clip(x, mi, ma):
    return min(max(mi, x), ma)


class EPcrust(object):
    '''Access EPcrust0.5x0.5 model.

        :param directory: Directory with the data files which contain the
            EPcrust model data: EPcrust_0_5.txt, Ice_thickness_0_5.txt
    '''

    fn_data = 'EPcrust_0_5.txt'
    fn_ice = 'Ice_thickness_0_5.txt'

    _instance = None

    def __init__(self):

        self.profile_keys = []
        self._typemap = None
        self._load_crustal_model()

    def get_profile(self, *args, **kwargs):
        '''
        Get crustal profile at a specific location or raw profile for given
        key.

        Get profile for location ``(lat, lon)``, or raw profile for given
        string key.

        :rtype: instance of :py:class:`EPcrust2Profile`
        '''

        lat = kwargs.pop('lat', None)
        lon = kwargs.pop('lon', None)

        if len(args) == 2:
            lat, lon = args

        if lat is not None and lon is not None:
            return self._typemap[self._indices(float(lat), float(lon))[0]]
        else:
            return self._raw_profiles[args[0]]

    def _indices(self, lat, lon):
        llat = int(lat * 2) / 2
        llon = int(lon * 2) / 2

        ilat = num.where(self._lonlat[:, 1] == llat)
        ilon = num.where(self._lonlat[:, 0] == llon)

        index = num.intersect1d(ilat, ilon)
        return index

    def _load_crustal_model(self):
        data = num.loadtxt(
            util.data_file(os.path.join('epcrust', EPcrust.fn_data)),
            skiprows=1)
        ice = num.loadtxt(
            util.data_file(os.path.join('epcrust', EPcrust.fn_ice)))

        ice = num.nan_to_num(ice)
        # Create profiles
        profiles = {}
        water_vp = 1.5
        water_vs = 0
        water_rho = 1.02

        ice_vp = 3.81
        ice_vs = 1.94
        ice_rho = 0.92

        for i in range(len(data)):
            if data[i, 2] < 0:
                water_thickness = abs(data[i, 2])
            else:
                water_thickness = 0

            thickness = num.concatenate(
                ([ice[i, 2], water_thickness], data[i, 3:6])) * 1000
            vp = num.concatenate(
                ([ice_vp, water_vp], data[i, 6:9])) * 1000
            vs = num.concatenate(
                ([ice_vs, water_vs], data[i, 9:12])) * 1000
            rho = num.concatenate(
                ([ice_rho, water_rho], data[i, 12:15])) * 1000
            elevation = data[i, 2] * 1000
            lon = data[i, 0]
            lat = data[i, 1]

            profiles[i] = EPcrust2Profile(
                i, lat, lon, vp, vs, rho, thickness, elevation)

        self._lonlat = data[:, :2]
        self._typemap = profiles
        self._raw_profiles = profiles
        self.profile_keys = sorted(profiles.keys())

    @staticmethod
    def instance():
        '''Get the default EPcrust instance.'''

        if EPcrust._instance is None:
            EPcrust._instance = EPcrust()

        return EPcrust._instance


def get_profile_keys():
    '''Get list of all profile keys.'''

    ep_crust = EPcrust.instance()
    return list(ep_crust.profile_keys)


def get_profile(*args, **kwargs):
    '''Get EPcrust 0.5x0.5 profile for given location or profile key.

    Get profile for (lat,lon) or raw profile for given string key.
    '''

    ep_crust = EPcrust.instance()
    return ep_crust.get_profile(*args, **kwargs)


def plot_crustal_thickness(ep_crust=None, filename='crustal_thickness.pdf'):
    '''
    Create a quick and dirty plot of the crustal thicknesses defined in
    EPcrust 0.5x0.5.
    '''

    if ep_crust is None:
        ep_crust = EPcrust.instance()

    def func(lat, lon):
        return ep_crust.get_profile(lat, lon).crustal_thickness(),

    plot(func, filename, zscaled_unit='km', zscaled_unit_factor=0.001)


def plot(func, filename, **kwargs):
    nlats, nlons = 162, 253
    lats = num.linspace(9, 89.5, nlats)
    lons = num.linspace(-56., 70., nlons)

    vecfunc = num.vectorize(func, [num.float])
    latss, lonss = num.meshgrid(lats, lons)
    thickness = vecfunc(latss, lonss)
    print(thickness)

    from pyrocko.plot import gmtpy
    cm = gmtpy.cm
    marg = (1.5*cm, 2.5*cm, 1.5*cm, 1.5*cm)
    p = gmtpy.Simple(
        width=20*cm, height=10*cm, margins=marg,
        with_palette=True, **kwargs)

    p.density_plot(gmtpy.tabledata(lons, lats, thickness.T))
    p.save(filename)


def avg_profile(lat, lon, radius, ep_crust=None):
    from pyrocko import orthodrome

    if ep_crust is None:
        ep_crust = EPcrust.instance()

    ex = orthodrome.radius_to_region(lat, lon, radius * 1000)
    extent = num.zeros(4)
    # round to closes EPcrust profile coordinate
    extent[0] = (int(ex[0])*2)/2
    extent[1] = (int(ex[1])*2)/2
    extent[2] = (int(ex[2])*2)/2
    extent[3] = (int(ex[3])*2)/2

    dlat = ((extent[3] - extent[2]) / 0.5) + 1
    dlon = ((extent[1] - extent[0]) / 0.5) + 1

    llat = num.linspace(extent[2], extent[3], dlat)
    llon = num.linspace(extent[0], extent[1], dlon)

    latss, lonss = num.meshgrid(llat, llon)

    elevation = num.zeros(int(dlat*dlon))
    vp = num.zeros((int(dlat*dlon), 5))
    vs = num.zeros((int(dlat*dlon), 5))
    rho = num.zeros((int(dlat*dlon), 5))
    thickness = num.zeros((int(dlat*dlon), 5))

    z = 0

    for i in range(int(dlon)):
        for j in range(int(dlat)):
            profile = ep_crust.get_profile(latss[i, j], lonss[i, j])
            elevation[z] = profile._elevation
            vp[z] = profile._vp
            vs[z] = profile._vs
            rho[z] = profile._rho
            thickness[z] = profile._thickness
            z = z + 1
    print(
        'Average EPcrust model from {} profiles '
        'in radius of {} km around'.format(int(dlat*dlon), radius,))

    profile = EPcrust2Profile(
        i,
        lat,
        lon,
        num.average(vp, 0),
        num.average(vs, 0),
        num.average(rho, 0),
        num.average(thickness, 0),
        num.average(elevation, 0))

    return profile
