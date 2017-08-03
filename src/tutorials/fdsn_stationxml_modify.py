import sys
from pyrocko.io import fdsn_station as fs

# load the StationXML data file passed
sx = fs.load_xml(filename=sys.argv[1])

comp_to_azi_dip = {
    'X': (0., 0.),
    'Y': (90., 0.),
    'Z': (0., -90.),
}

# step through all the networks within the data file
for network in sx.network_list:

    # step through all the stations per networks
    for station in network.station_list:

        # step through all the channels per stations
        for channel in station.channel_list:
            azi, dip = comp_to_azi_dip[channel.code]

            # change the azimuth and dip of the channel per channel alpha
            # code
            channel.azimuth.value = azi
            channel.dip.value = dip

            # set the instrument input units to 'M'eters
            channel.response.instrument_sensitivity.input_units.name = 'M'

# save as new StationXML file
sx.dump_xml(filename='changed.xml')
