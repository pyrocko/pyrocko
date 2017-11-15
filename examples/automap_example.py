from pyrocko import model
from pyrocko import gmtpy
from pyrocko.plot.automap import Map
from pyrocko.example import get_example_data


gmtpy.check_have_gmt()

# Download example data
get_example_data('stations_deadsea.pf')

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
for i in range(len(stations)):
    m.add_label(lats[i], lons[i], labels[i])

# Draw a beachball
#m.gmt.psmeca(
#    S='m.5', G='red', C='5p,0/0/0', in_rows=[
        # location and moment tensor components (from www.globalcmt.org)
#        (35.31, 31.62, 10, -0.27, 0.53, -0.27, -0.66, -0.35, -0.74, 24,
#         35.31, 31.62, 'Event - 2004/01/11'),
#    ], *m.jxyr)

# Load events from catalog file (generated using catalog.GlobalCMT() to download from www.globalcmt.org)
# If no moment tensor is provided in the catalogue, the event is plotted as a red circle.
# Symbol size relative to magnitude.
events = model.load_events('deadsea_gcmt.txt')
print(events)
beachball_symbol = 'd'
for ev in events:
    if ev.moment_tensor is None:
        mag = ev.magnitude
        ev_size = 'c'+str(mag*5.)+'p' 
        m.gmt.psxy(
            in_rows=[[ev.lon, ev.lat]],
            S=ev_size,
            G=gmtpy.color('scarletred2'),
            W='1p,black',
            *m.jxyr)

    else:
        devi = ev.moment_tensor.deviatoric()
        mag = ev.magnitude
        beachball_size = mag*5.0
        mt = devi.m_up_south_east()
        mt = mt / ev.moment_tensor.scalar_moment() \
            * pmt.magnitude_to_moment(5.0)
        m6 = pmt.to6(mt)
        data = (ev.lon, ev.lat, 10) + tuple(m6) + (1, 0, 0)#

        if m.gmt.is_gmt5():
            kwargs = dict(
                M=True,
                S='%s%g' % (beachball_symbol[0], (beachball_size) / gmtpy.cm))
        else:
            kwargs = dict(
                S='%s%g' % (beachball_symbol[0], (beachball_size)*2 / gmtpy.cm))

        m.gmt.psmeca(
            in_rows=[data],
            G=gmtpy.color('chocolate1'),
            E='white',
            W='1p,%s' % gmtpy.color('chocolate3'),
            *m.jxyr,
            **kwargs)

m.save('automap_deadsea.png')
