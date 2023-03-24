from pyrocko import orthodrome

# arguments: origin lat, origin lon, north [m], east [m]
lat, lon = orthodrome.ne_to_latlon(10.3, 12.4, 22200., 21821.)

print('latitude: %s, longitude: %s ' % (lat, lon))

# >>> latitude: 10.4995878932, longitude: 12.5995823469
