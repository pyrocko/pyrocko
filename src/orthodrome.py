import math
import numpy as num
from pyrocko.config import config

d2r = math.pi/180.
r2d = 1./d2r
earth_oblateness = 1./298.257223563
earthradius_equator = 6378.14 * 1000.
earthradius = config().earthradius
d2m = earthradius_equator*math.pi/180.
m2d = 1./d2m

class Loc:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

def clip(x, mi, ma):
    return num.minimum(num.maximum(mi,x),ma)
        
def wrap(x, mi, ma):
    return x - num.floor((x-mi)/(ma-mi)) * (ma-mi)

def cosdelta(a, b):
    return min(1.0, math.sin(a.lat*d2r) * math.sin(b.lat*d2r) + math.cos(a.lat*d2r) * math.cos(b.lat*d2r) * math.cos(d2r*(b.lon-a.lon)))
    
def cosdelta_numpy(a_lats, a_lons, b_lats, b_lons):
    return num.minimum(1.0, num.sin(a_lats*d2r) * num.sin(b_lats*d2r) + num.cos(a_lats*d2r) * num.cos(b_lats*d2r) * num.cos(d2r*(b_lons-a_lons)))

def azimuth(a, b):
    return r2d*math.atan2( math.cos(a.lat*d2r) * math.cos(b.lat*d2r) * math.sin(d2r*(b.lon-a.lon)),
                           math.sin(d2r*b.lat) - math.sin(d2r*a.lat) * cosdelta(a,b) )
                           
def azimuth_numpy(a_lats, a_lons, b_lats, b_lons, _cosdelta=None):
    if _cosdelta == None:
        _cosdelta = cosdelta_numpy(a_lats,a_lons,b_lats,b_lons)
        
    return r2d*num.arctan2( num.cos(a_lats*d2r) * num.cos(b_lats*d2r) * 
                            num.sin(d2r*(b_lons-a_lons)),
                            num.sin(d2r*b_lats) - num.sin(d2r*a_lats) * _cosdelta )

def azidist_numpy(*args):
    _cosdelta = cosdelta_numpy(*args)
    _azimuths = azimuth_numpy( _cosdelta=_cosdelta,*args)
    return _azimuths, r2d*num.arccos(_cosdelta)

def distance_accurate50m( a, b ):

    # more accurate distance calculation based on a spheroid of rotation

    # returns distance in [m] between points a and b
    # coordinates must be given in degrees

    # should be accurate to 50 m using WGS84
    
    # from wikipedia :  http://de.wikipedia.org/wiki/Orthodrome
    # based on: Meeus, J.: Astronomical Algorithms, S 85, Willmann-Bell,
    #           Richmond 2000 (2nd ed., 2nd printing), ISBN 0-943396-61-1
    
    
    f = (a.lat + b.lat)*d2r / 2.
    g = (a.lat - b.lat)*d2r / 2.
    l = (a.lon - b.lon)*d2r / 2.

    s = math.sin(g)**2 * math.cos(l)**2 + math.cos(f)**2 * math.sin(l)**2
    c = math.cos(g)**2 * math.cos(l)**2 + math.sin(f)**2 * math.sin(l)**2

    w = math.atan( math.sqrt( s/c ) )

    if w == 0.0:
        return 0.0

    r = math.sqrt(s*c)/w
    d = 2.*w*earthradius_equator
    h1 = (3.*r-1.)/(2.*c)
    h2 = (3.*r+1.)/(2.*s)

    return d * (1.+ earth_oblateness * h1 * math.sin(f)**2 * math.cos(g)**2 - 
                    earth_oblateness * h2 * math.cos(f)**2 * math.sin(g)**2)

def distance_accurate50m_numpy( a_lats, a_lons, b_lats, b_lons ):
    # same as distance_accurate50m, but using numpy arrays

    eq = num.logical_and(a_lats == b_lats, a_lons == b_lons)
    ii_neq = num.where(num.logical_not(eq))[0]

    if num.all(eq):
        return num.zeros_like(eq, dtype=num.float)

    def extr(x):
        if isinstance(x, num.ndarray) and x.size > 1:
            return x[ii_neq]
        else:
            return x

    a_lats = extr(a_lats)
    a_lons = extr(a_lons)
    b_lats = extr(b_lats)
    b_lons = extr(b_lons)

    f = (a_lats + b_lats)*d2r / 2.
    g = (a_lats - b_lats)*d2r / 2.
    l = (a_lons - b_lons)*d2r / 2.

    s = num.sin(g)**2 * num.cos(l)**2 + num.cos(f)**2 * num.sin(l)**2
    c = num.cos(g)**2 * num.cos(l)**2 + num.sin(f)**2 * num.sin(l)**2

    w = num.arctan( num.sqrt( s/c ) )

    r = num.sqrt(s*c)/w

    d = 2.*w*earthradius_equator
    h1 = (3.*r-1.)/(2.*c)
    h2 = (3.*r+1.)/(2.*s)

    dists = num.zeros(eq.size, dtype=num.float)
    dists[ii_neq] = d * (1.+ earth_oblateness * h1 * num.sin(f)**2 *
        num.cos(g)**2 - earth_oblateness * h2 * num.cos(f)**2 * num.sin(g)**2)

    return dists

def ne_to_latlon( lat0, lon0, north_m, east_m ):
    '''Transform local carthesian coordinates to latitude and longitude.
    
    lat0, lon0:      Origin of the carthesian coordinate system.
    north_m, east_m: 1D numpy arrays with distances from origin in meters.
    
    Returns: lat, lon: 1D numpy arrays with latitudes and longitudes
    
    The projection used preserves the azimuths of the input points.
    '''
    
    a = num.sqrt(north_m**2+east_m**2)/config().earthradius
    gamma = num.arctan2(east_m,north_m)
    
    return azidist_to_latlon_rad( lat0, lon0, gamma, a)

def azidist_to_latlon(lat0, lon0, azimuth_deg, distance_deg):
    return azidist_to_latlon_rad( lat0, lon0, azimuth_deg/180.*num.pi, distance_deg/180.*num.pi)

def azidist_to_latlon_rad( lat0, lon0, azimuth_rad, distance_rad):
    
    a = distance_rad
    gamma = azimuth_rad

    b = math.pi/2.-lat0*d2r
    
    alphasign = 1.
    alphasign = num.where(gamma < 0, -1., 1.)
    gamma = num.abs(gamma)
    
    c = num.arccos( clip(num.cos(a)*num.cos(b)+num.sin(a)*num.sin(b)*num.cos(gamma),-1.,1.) )
    alpha = num.arcsin( clip(num.sin(a)*num.sin(gamma)/num.sin(c),-1.,1.) )
    
    alpha = num.where(num.cos(a)-num.cos(b)*num.cos(c) < 0, 
        num.where(alpha > 0,  math.pi-alpha, -math.pi-alpha), alpha)
    
    lat = r2d * (math.pi/2. - c)
    lon = wrap(lon0 + r2d*alpha*alphasign,-180.,180.)
    
    return lat, lon


def ne_to_latlon_alternative_method( lat0, lon0, north_m, east_m ):

    '''Like ne_to_latlon(), but this method, although it should be numerically
    more stable, suffers problems at points which are 'across the pole' as seen
    from the carthesian origin.'''

    b = math.pi/2.-lat0*d2r
    a = num.sqrt(north_m**2+east_m**2)/config().earthradius

    
    gamma = num.arctan2(east_m,north_m)
    alphasign = 1.
    alphasign = num.where(gamma < 0., -1., 1.)
    gamma = num.abs(gamma)
    
    z1 = num.cos((a-b)/2.)*num.cos(gamma/2.)
    n1 = num.cos((a+b)/2.)*num.sin(gamma/2.)
    z2 = num.sin((a-b)/2.)*num.cos(gamma/2.)
    n2 = num.sin((a+b)/2.)*num.sin(gamma/2.)
    t1 = num.arctan2( z1,n1 )
    t2 = num.arctan2( z2,n2 )
    
    alpha = t1 + t2
    beta  = t1 - t2
    
    sin_t1 = num.sin(t1)
    sin_t2 = num.sin(t2)           
    c = num.where( num.abs(sin_t1)>num.abs(sin_t2), 
                num.arccos(z1/sin_t1)*2.,
                num.arcsin(z2/sin_t2)*2. )
            
    lat = r2d * (math.pi/2. - c)
    lon = wrap(lon0 + r2d*alpha*alphasign,-180.,180.)
    return lat, lon

def latlon_to_ne(refloc, loc):
    azi = azimuth(refloc, loc)
    dist = distance_accurate50m(refloc, loc)
    n, e = math.cos(azi*d2r)*dist, math.sin(azi*d2r)*dist
    return n,e

def latlon_to_ne_numpy(lat0, lon0, lat, lon):
    azi = azimuth_numpy(lat0, lon0, lat, lon)
    dist = distance_accurate50m_numpy(lat0, lon0, lat, lon)
    n = num.cos(azi*d2r)*dist
    e = num.sin(azi*d2r)*dist
    return n, e

_wgs84 = None
def get_wgs84():
    global _wgs84
    if _wgs84 is None:
        from geographiclib.geodesic import Geodesic
        _wgs84 = Geodesic.WGS84

    return _wgs84

def amap(n):
    def wrap(f):
        if n == 1:
            def func(*args):
                it = num.nditer(args + (None,))
                for ops in it:
                    ops[-1][...] = f(*ops[:-1])

                return it.operands[-1]
        elif n == 2:
            def func(*args):
                it = num.nditer(args + (None, None))
                for ops in it:
                    ops[-2][...], ops[-1][...] = f(*ops[:-2])

                return it.operands[-2], it.operands[-1]

        return func
    return wrap

@amap(2)
def ne_to_latlon2(lat0, lon0, north_m, east_m):
    wgs84 = get_wgs84()
    az = num.arctan2(east_m, north_m)*r2d
    dist = num.sqrt(east_m**2 + north_m**2)
    x = wgs84.Direct(lat0, lon0, az, dist)
    return x['lat2'], x['lon2']

@amap(2)
def latlon_to_ne2(lat0, lon0, lat1, lon1):
    wgs84 = get_wgs84()
    x = wgs84.Inverse(lat0, lon0, lat1, lon1)
    dist = x['s12']
    az = x['azi1']
    n = num.cos(az*d2r)*dist
    e = num.sin(az*d2r)*dist
    return n, e

@amap(1)
def distance_accurate15nm(lat1, lon1, lat2, lon2):
    wgs84 = get_wgs84()
    return wgs84.Inverse(lat1, lon1, lat2, lon2)['s12']


def positive_region(region):
    west, east, south, north = [float(x) for x in region]

    assert -180. - 360. <= west < 180.
    assert -180. < east <= 180. + 360.
    assert -90. <= south < 90.
    assert -90. < north <= 90.

    if east < west:
        east += 360.

    if west < -180.:
        west += 360.
        east += 360.

    return (west, east, south, north)


def radius_to_region(lat, lon, radius):
    radius_deg = radius * m2d
    if radius_deg < 45.:
        lat_min = max(-90., lat - radius_deg)
        lat_max = min(90., lat + radius_deg)
        absmaxlat = max(abs(lat_min), abs(lat_max))
        if absmaxlat > 89:
            lon_min = -180.
            lon_max = 180.
        else:
            lon_min = max(
                -180. - 360.,
                lon - radius_deg / math.cos(absmaxlat*d2r))
            lon_max = min(
                180. + 360.,
                lon + radius_deg / math.cos(absmaxlat*d2r))

        lon_min, lon_max, lat_min, lat_max = positive_region(
            (lon_min, lon_max, lat_min, lat_max))

        return lon_min, lon_max, lat_min, lat_max

    else:
        return None
