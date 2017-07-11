import numpy as num
from pyrocko.gshhg import GSHHG

gshhg = GSHHG.intermediate()
# gshhg = GSHHG.full()
# gshhg = GSHHG.low()

gshhg.is_point_on_land(lat=32.1, lon=44.2)


# Create a landmask for a regular grid
lats = num.linspace(30., 50., 100)
lons = num.linspace(2., 10., 100)

lat_grid, lon_grid = num.meshgrid(lats, lons)
coordinates = num.array([lat_grid.ravel(), lon_grid.ravel()]).T

land_mask = gshhg.get_land_mask(coordinates).reshape(*lat_grid.shape)
