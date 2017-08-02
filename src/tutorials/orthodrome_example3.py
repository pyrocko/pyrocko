from pyrocko import orthodrome

# For a single point
orthodrome.azibazi(49.1, 20.5, 45.4, 22.3)

# >>> (161.05973376168285, -17.617746351508035)  # Azimuth and backazimuth

import numpy as num

ncoords = 1000
# First set of coordinates
lats_a = num.random.random_integers(-180, 180, ncoords)
lons_a = num.random.random_integers(-90, 90, ncoords)

# Second set of coordinates
lats_b = num.random.random_integers(-180, 180, ncoords)
lons_b = num.random.random_integers(-90, 90, ncoords)

orthodrome.azibazi_numpy(lats_a, lons_a, lats_b, lons_b)
