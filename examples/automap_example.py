from pyrocko.plot.automap import Map
from pyrocko.example import get_example_data
from pyrocko import model
from pyrocko import moment_tensor as pmt
from pyrocko.plot import gmtpy

gmtpy.check_have_gmt()

# Download example data
get_example_data('stations_deadsea.pf')
get_example_data('deadsea_events_1976-2017.txt')

# Generate the basic map
m = Map(
    lat=31.5,
    lon=35.5,
    radius=150000.,
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


# Load events from catalog file (generated using catalog.GlobalCMT()
# download from www.globalcmt.org)
# If no moment tensor is provided in the catalogue, the event is plotted
# as a red circle. Symbol size relative to magnitude.

events = model.load_events('deadsea_events_1976-2017.txt')
beachball_symbol = 'd'
factor_symbl_size = 5.0
for ev in events:
    mag = ev.magnitude
    if ev.moment_tensor is None:
        ev_symb = 'c'+str(mag*factor_symbl_size)+'p'
        m.gmt.psxy(
            in_rows=[[ev.lon, ev.lat]],
            S=ev_symb,
            G=gmtpy.color('scarletred2'),
            W='1p,black',
            *m.jxyr)
    else:
        devi = ev.moment_tensor.deviatoric()
        beachball_size = mag*factor_symbl_size
        mt = devi.m_up_south_east()
        mt = mt / ev.moment_tensor.scalar_moment() \
            * pmt.magnitude_to_moment(5.0)
        m6 = pmt.to6(mt)
        data = (ev.lon, ev.lat, 10) + tuple(m6) + (1, 0, 0)

        if m.gmt.is_gmt5():
            kwargs = dict(
                M=True,
                S='%s%g' % (beachball_symbol[0], (beachball_size) / gmtpy.cm))
        else:
            kwargs = dict(
                S='%s%g' % (beachball_symbol[0],
                            (beachball_size)*2 / gmtpy.cm))

        m.gmt.psmeca(
            in_rows=[data],
            G=gmtpy.color('chocolate1'),
            E='white',
            W='1p,%s' % gmtpy.color('chocolate3'),
            *m.jxyr,
            **kwargs)

m.save('automap_deadsea.png')
