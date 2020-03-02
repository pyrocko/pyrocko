from pyrocko.io import stationxml
from pyrocko.example import get_example_data

# Download example StationXML file
get_example_data('responses.xml')

# load the StationXML downloaded data file
sx = stationxml.load_xml(filename='responses.xml')

# Extract Station objects from FDSNStationXML object
pyrocko_stations = sx.get_pyrocko_stations()
