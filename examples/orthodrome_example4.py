from pyrocko import orthodrome

# option 1, coordinates as floats
north_m, east_m = orthodrome.latlon_to_ne(
    10.3,   # origin latitude
    12.4,   # origin longitude
    10.5,   # target latitude
    12.6)   # target longitude

print(north_m, east_m)

# >>> 22199.7843582 21821.3511789

# option 2, coordinates from instances with 'lon' and 'lat' attributes

from pyrocko.gf import seismosizer   # noqa

source = seismosizer.DCSource(lat=10.3, lon=12.4)
target = seismosizer.Target(lat=10.5, lon=12.6)

north_m, east_m = orthodrome.latlon_to_ne(source, target)

print(north_m, east_m)

# >>> 22199.7843582 21821.3511789
