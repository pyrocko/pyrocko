import numpy as num
from pyrocko.dataset.gshhg import GSHHG
from matplotlib import pyplot as plt

gshhg = GSHHG.intermediate()
# gshhg = GSHHG.full()
# gshhg = GSHHG.low()

gshhg.is_point_on_land(lat=32.1, lon=44.2)


# Create a landmask for a regular grid
lats = num.linspace(30., 50., 500)
lons = num.linspace(2., 10., 500)

lon_grid, lat_grid = num.meshgrid(lons, lats)
coordinates = num.array([lat_grid.ravel(), lon_grid.ravel()]).T

land_mask = gshhg.get_land_mask(coordinates).reshape(*lat_grid.shape)

plt.pcolormesh(lons, lats, land_mask, cmap='Greys', shading='nearest')
plt.show()
