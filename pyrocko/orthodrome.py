import math
import numpy as num
import config


d2r = math.pi/180.
r2d = 1./d2r
earth_oblateness = 1./298.257223563
earthradius_equator = 6378.14 * 1000.

def clip(x, mi, ma):
    return num.minimum(num.maximum(mi,x),ma)
        
def wrap(x, mi, ma):
    return x - num.floor((x-mi)/(ma-mi)) * (ma-mi)

def cosdelta(a, b):
    return math.sin(a.lat*d2r) * math.sin(b.lat*d2r) + math.cos(a.lat*d2r) * math.cos(b.lat*d2r) * math.cos(d2r*(b.lon-a.lon))
    
def cosdelta_numpy(a_lats, a_lons, b_lats, b_lons):
    return num.sin(a_lats*d2r) * num.sin(b_lats*d2r) + num.cos(a_lats*d2r) * num.cos(b_lats*d2r) * num.cos(d2r*(b_lons-a_lons))

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
    r = math.sqrt(s*c)/w
    d = 2.*w*earthradius_equator
    h1 = (3.*r-1.)/(2.*c)
    h2 = (3.*r+1.)/(2.*s)

    return d * (1.+ earth_oblateness * h1 * math.sin(f)**2 * math.cos(g)**2 - 
                    earth_oblateness * h2 * math.cos(f)**2 * math.sin(g)**2)
                    
def distance_accurate50m_numpy( a_lats, a_lons, b_lats, b_lons ):

    # same as distance_accurate50m, but using numpy arrays
    
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

    return d * (1.+ earth_oblateness * h1 * num.sin(f)**2 * num.cos(g)**2 - 
                    earth_oblateness * h2 * num.cos(f)**2 * num.sin(g)**2)

def ne_to_latlon( lat0, lon0, north_m, east_m ):
    '''Transform local carthesian coordinates to latitude and longitude.
    
    lat0, lon0:      Origin of the carthesian coordinate system.
    north_m, east_m: 1D numpy arrays with distances from origin in meters.
    
    Returns: lat, lon: 1D numpy arrays with latitudes and longitudes
    
    The projection used preserves the azimuths of the input points.
    '''
    
    a = num.sqrt(north_m**2+east_m**2)/config.earthradius
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
    
    c = num.arccos( clip(num.cos(a)*math.cos(b)+num.sin(a)*math.sin(b)*num.cos(gamma),-1.,1.) )
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
    a = num.sqrt(north_m**2+east_m**2)/config.earthradius

    
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


