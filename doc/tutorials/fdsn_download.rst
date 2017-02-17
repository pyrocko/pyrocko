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
