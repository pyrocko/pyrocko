import numpy as num
from pyrocko import orthodrome

ncoords = 1000

# First set of coordinates
lats_a = num.random.random_integers(-180, 180, ncoords)
lons_a = num.random.random_integers(-90, 90, ncoords)

# Second set of coordinates
lats_b = num.random.random_integers(-180, 180, ncoords)
lons_b = num.random.random_integers(-90, 90, ncoords)

orthodrome.distance_accurate50m_numpy(lats_a, lons_a, lats_b, lons_b)
