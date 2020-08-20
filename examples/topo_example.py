import numpy as num
from pyrocko import topo, plot, orthodrome as od

lon_min, lon_max, lat_min, lat_max = 14.34, 14.50, 40.77, 40.87
dem_name = 'SRTMGL3'

# extract gridded topography (possibly downloading first)
tile = topo.get(dem_name, (lon_min, lon_max, lat_min, lat_max))

# geographic to local cartesian coordinates
lons = tile.x()
lats = tile.y()
lons2 = num.tile(lons, lats.size)
lats2 = num.repeat(lats, lons.size)
norths, easts = od.latlon_to_ne_numpy(lats[0], lons[0], lats2, lons2)
norths = norths.reshape((lats.size, lons.size))
easts = easts.reshape((lats.size, lons.size))

# plot it
plt = plot.mpl_init(fontsize=10.)
fig = plt.figure(figsize=plot.mpl_papersize('a5', 'landscape'))
axes = fig.add_subplot(1, 1, 1, aspect=1.0)
cbar = axes.pcolormesh(easts, norths, tile.data,
                       cmap='gray', shading='gouraud')
fig.colorbar(cbar).set_label('Altitude [m]')
axes.set_title(dem_name)
axes.set_xlim(easts.min(), easts.max())
axes.set_ylim(norths.min(), norths.max())
axes.set_xlabel('Easting [m]')
axes.set_ylabel('Northing [m]')
fig.savefig('topo_example.png')
