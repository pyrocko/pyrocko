Downloading seismic data (FDSN)
================================

Waveforms and meta data can be retrieved from online `FDSN services <http://www.fdsn.org>`_ using the :py:class:`pyrocko.fdsn` modules.


Seismic data from Geofon
-------------------------

The following demo explains how to download both, waveform, as well as response information. Latter is used to deconvolve the transfer function from traces in a second step.

::

    from pyrocko.fdsn import ws
    from pyrocko import util, io, trace

    tmin = util.stt('2014-01-01 16:10:00.000')
    tmax = util.stt('2014-01-01 16:39:59.000')

    # select stations by their NSLC id and wildcards (asterik)
    selection = [
        ('*', 'HMDT', '*', '*', tmin, tmax),    # all available components
        ('GE', 'EIL', '*', '*Z', tmin, tmax),   # all vertical components
    ]

    # setup a waveform data request
    request_waveform = ws.dataselect(site='geofon', selection=selection)

    # write the incoming data stream to 'traces.mseed'
    with open('traces.mseed', 'w') as file:
        file.write(request_waveform.read())

    # request meta data
    request_response = ws.station(
        site='geofon', selection=selection, level='response')

    # save the response in yaml format
    request_response.dump(filename='responses.yaml')

    # Loop through retrieved waveforms and request meta information
    # for each trace
    traces = io.load('traces.mseed')
    displacement = []
    for tr in traces:
        polezero_response = request_response.get_pyrocko_response(
            nslc=tr.nslc_id,
            timespan=(tr.tmin, tr.tmax),
            fake_input_units='M')       # (Required for consistent responses
                                        # throughout entire data set)

        # deconvolve transfer function
        restituted = tr.transfer(
            tfade=2.,
            freqlimits=(0.01, 0.1, 1., 2.),
            transfer_function=polezero_response,
            invert=True)

        displacement.append(restituted)

    # scrutinize displacement traces
    # Inspect using the snuffler
    trace.snuffle(displacement)


StationXML data manipulation
----------------------------

To manipulate `StationXML <http://www.fdsn.org/xml/station/>`_ data through Pryocko use the
:py:mod:`pyrocko.fdsn.station` module.  This exmaple will change the azimuth
and dip values for channels whose codes are X, Y and Z, and set all channel
instrument's input units to meters.

::

    import sys
    from pyrocko.fdsn import station as fs

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


Manipulation of downloaded StationXML data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A combination of the two sections above, allows one to easily change the azimuth, 
dip and instrument units of the downloaded data before saving as a
`StationXML <http://www.fdsn.org/xml/station/>`_ file.

::

    from pyrocko.fdsn import ws
    from pyrocko import util, io, trace

    tmin = util.stt('2014-01-01 16:10:00.000')
    tmax = util.stt('2014-01-01 16:39:59.000')

    # select stations by their NSLC id and wildcards (asterik)
    selection = [
        ('*', 'HMDT', '*', '*', tmin, tmax),    # all available components
        ('GE', 'EIL', '*', '*Z', tmin, tmax),   # all vertical components
    ]

    # setup a waveform data request
    request_waveform = ws.dataselect(site='geofon', selection=selection)

    # request meta data
    request_response = ws.station(
        site='geofon', selection=selection, level='response', format='xml')

    comp_to_azi_dip = {
        'N': (0., 0.),
        'E': (90., 0.),
        'Z': (0., -90.),
    }

    # step through all the networks within the data file
    for network in request_response.network_list:

        # step through all the stations per networks
        for station in network.station_list:

            # step through all the channels per stations
            for channel in station.channel_list:

                # get the physical orientation of the sensor
                chcode = channel.code[-1]

                if chcode in comp_to_azi_dip.keys():
                    azi, dip = comp_to_azi_dip[chcode]

                    # change the azimuth and dip of the channel per channel alpha
                    # code
                    channel.azimuth.value = azi
                    channel.dip.value = dip

                # set the instrument input units to 'M'eters
                channel.response.instrument_sensitivity.input_units.name = 'M'

    request_response.dump_xml(filename='download_changed.xml')
