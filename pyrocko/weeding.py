import math, time
import numpy as num
from pyrocko import orthodrome

def neighborhood_density(dists, neighborhood=1):
    sdists = dists.copy()
    sdists.sort(axis=1)
    meandists = num.mean(sdists[:,1:1+neighborhood],axis=1)
    return meandists
    

def _weed(dists, badnesses, neighborhood=1, interaction_radius=3., del_frac=4, max_del=100, max_depth=100, depth=0):
    if depth > max_depth:
        assert False, 'max recursion depth reached'
    
    meandists = neighborhood_density(dists, neighborhood)
    
    order = meandists.argsort()
    candidates = order[:order.size/del_frac+1]
    badness_candidates = badnesses[candidates]
    order_badness = (-badness_candidates).argsort()
    
    order[:order.size/del_frac+1] = order[:order.size/del_frac+1][order_badness]
    
    deleted = num.zeros(order.size, dtype=num.bool)
    ndeleted = 0
    for i,ind in enumerate(order):
        if (i<order.size/del_frac/2+1 and 
            ndeleted < max_del and 
                num.all(dists[ind,deleted] > interaction_radius*meandists[ind])):
            deleted[ind] = True
            ndeleted += 1
           
    if ndeleted == 0:
        return deleted
    
    kept = num.logical_not(deleted).nonzero()[0]
    
    xdists = dists[num.meshgrid(kept,kept)]
    xbadnesses = badnesses[kept]
    xdeleted = weed(xdists, xbadnesses, neighborhood, interaction_radius, del_frac, 
                                max_del-ndeleted, max_depth, depth+1)
    
    deleted[kept] = xdeleted
    return deleted

def weed(x, y, badnesses, neighborhood=1, nwanted=None, interaction_radius=3.):
    assert x.size == y.size
    n = x.size
    NEW = num.newaxis
    ax = x[NEW,:]
    ay = y[NEW,:]
    bx = x[:,NEW]
    by = y[:,NEW]
    
    if nwanted is None:
        nwanted = n/2
    
    dx = num.abs(ax-bx)
    dx = num.where(dx > 180., 360.-dx, dx)
    
    dists = num.sqrt(dx**2 + (ay-by)**2)
    deleted = _weed(dists, badnesses, neighborhood, interaction_radius, del_frac=4,
                              max_del=n-nwanted, max_depth=500, depth=0)    

    kept = num.logical_not(deleted).nonzero()[0]
    dists_kept = dists[num.meshgrid(kept,kept)]
    meandists_kept = neighborhood_density(dists_kept, neighborhood)
    return deleted, meandists_kept

def weed_stations(event, stations, nwanted, badnesses=None, neighborhood=3):
    
    lats = [ station.lat for station in stations ]
    lons = [ station.lon for station in stations ]
    azimuths, dists = orthodrome.azidist_numpy(event.lat, event.lon, lats, lons)
    if badnesses is None:
        badnesses = num.ones(len(lats),dtype=float)
    
    deleted, meandists_kept = station_selector(azimuths, dists, badnesses, nwanted=nwanted, neighborhood=neighborhood)
    
    stations_weeded = [ station for (delete, station) in zip(deleted, stations) if not delete ]
    return stations_weeded

