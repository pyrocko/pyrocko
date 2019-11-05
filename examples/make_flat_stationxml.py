from pyrocko.io import stationxml as fdsn
from pyrocko import model
from pyrocko.example import get_example_data

get_example_data('stations.txt')

stations = model.load_stations('stations.txt')

station_xml = fdsn.FDSNStationXML.from_pyrocko_stations(stations)
for network in station_xml.network_list:
    for station in network.station_list:
        for channel in station.channel_list:
            channel.response = fdsn.Response(
                instrument_sensitivity=fdsn.Sensitivity(
                    value=1.0,
                    frequency=1.0,
                    input_units=fdsn.Units('M'),
                    output_units=fdsn.Units('COUNTS')))
                    
station_xml.validate()
# print(station_xml.dump_xml())
station_xml.dump_xml(filename='stations_flat_displacement.xml')
