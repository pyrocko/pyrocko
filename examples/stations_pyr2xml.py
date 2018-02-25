from pyrocko.io import stationxml
from pyrocko import model
from pyrocko.example import get_example_data


# get example data
station_file = get_example_data('stations.txt')

# load pyrocko stations
stations = model.station.load_stations(station_file)

# get station xml from pyrocko stations
st_xml = stationxml.FDSNStationXML.from_pyrocko_stations(stations)

# save stations as xml file
st_xml.dump_xml(filename='stations.xml')
