from pyrocko import pile, io, util, model
from pyrocko.tutorials.util import get_tutorial_data


# Download example data
get_tutorial_data('data_conversion')



input_path = 'data_conversion/mseed'
output_path = 'data_conversion/sac/' \
        '%(dirhz)s/%(station)s_%(channel)s_%(tmin)s.sac'

fn_stations = 'data_conversion/stations.txt'

stations_list = model.load_stations(fn_stations)

stations = {}
for s in stations_list:
    stations[s.network, s.station, s.location] = s
    s.set_channels_by_name(*'BHN BHE BHZ BLN BLE BLZ'.split())

p = pile.make_pile(input_path)
h = 3600.
tinc = 1*h
tmin = util.day_start(p.tmin)
for traces in p.chopper_grouped(tmin=tmin, tinc=tinc,
                                gather=lambda tr: tr.nslc_id):
    for tr in traces:
        dirhz = '%ihz' % int(round(1./tr.deltat))
        io.save(
            [tr], output_path,
            format='sac',
            additional={'dirhz': dirhz},
            stations=stations
        )
