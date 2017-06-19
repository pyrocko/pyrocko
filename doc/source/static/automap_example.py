from pyrocko import model
from pyrocko.automap import Map

# Generate the basic map
m = Map(
    lat=31.5,
    lon=35.5,
    radius=100000.,
    width=30., height=30.,
    show_grid=False,
    show_topo=True,
    color_dry=(238, 236, 230),
    topo_cpt_wet='light_sea_uniform',
    topo_cpt_dry='light_land_uniform',
    illuminate=True,
    illuminate_factor_ocean=0.15,
    show_rivers=False,
    show_plates=True)

# Draw some larger cities covered by the map area
m.draw_cities()

# Generate with latitute, longitude and labels of the stations
stations = model.load_stations('stations_deadsea.pf')
lats = [s.lat for s in stations]
lons = [s.lon for s in stations]
labels = ['.'.join(s.nsl()) for s in stations]

# Stations as black triangles. Genuine GMT commands can be parsed by the maps'
# gmt attribute. Last argument of the psxy function call pipes the maps'
# pojection system.
m.gmt.psxy(in_columns=(lons, lats), S='t20p', G='black', *m.jxyr)

# Station labels
for i in xrange(len(stations)):
    m.add_label(lats[i], lons[i], labels[i])

# Draw a beachball
m.gmt.psmeca(
    S='m.5', G='red', C='5p,0/0/0', in_rows=[
        # location and moment tensor components (from www.globalcmt.org)
        (35.31, 31.62, 10, -0.27, 0.53, -0.27, -0.66, -0.35, -0.74, 24,
         35.31, 31.62, 'Event - 2004/01/11'),
    ], *m.jxyr)

m.save('automap_deadsea.png')
