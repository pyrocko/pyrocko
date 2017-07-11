Dataset management
==================

At the base of Pyrocko's waveform dataset handling is the
:class:`~pyrocko.pile.Pile` class. It organizes and caches the meta-data of
large waveform datasets split into many files and provides on-demand loading of
waveform data so that only data relevant to the current process/view has to be
read from disk into the limited computer memory. It can efficiently handle up
to a few 100000 files for interactive processes.

Reorganizing a dataset into hour-files
--------------------------------------
In each iteration we get all data for the current time window as a list of traces. The traces emitted by :py:meth:`pyrocko.pile.Pile.chopper()` 'know' the time window to which they belong; it is stored in the attributes ``trace.wmin`` and ``trace.wmax``.
note: ``trace.tmin`` (its onset) does not have to be identical to ``trace.wmin``. The directory parts in the output path will be created as neccessary.
When applying this procedure to a dataset consisting of arbitrarily separated files, it will automatically connect adjacent traces as needed!
The :mod:`time` and :mod:`calendar` modules will be used. 
::

    from pyrocko import pile, io, util
    import time, calendar 
    
    p = pile.make_pile(['test.mseed'])  # could give directories or thousands of filenames here
    
    # get timestamp for full hour before first data sample in all selected traces
    tmin = calendar.timegm( time.gmtime(p.tmin)[:4] + ( 0, 0 ) )
    
    # iterate over the data, with a window length of one hour
    for traces in p.chopper(tmin=tmin, tinc=3600):
        if traces:    # the list could be empty due to gaps
            window_start = traces[0].wmin
            timestring = util.time_to_str(window_start, format='%Y-%m-%d_%H')
            filepath = 'test_hourfiles/hourfile-%s.mseed' % timestring
            io.save(traces, filepath)
            
            
            
            
Downsampling a whole dataset
----------------------------

Example for downsampling all trace files in the input folder to a common sampling rate with :py:meth:`pyrocko.trace.Trace.downsample_to`.

::

    from pyrocko import pile, io, util
    import time, calendar

    # when pile.make_pile() is called without any arguments, the command line
    # parameters given to the script are searched for waveform files and directories
    p = pile.make_pile()

    # get timestamp for full hour before first data sample in all selected traces
    tmin = calendar.timegm( time.gmtime(p.tmin)[:4] + ( 0, 0 ) )

    tinc = 3600.
    tpad = 10.
    target_deltat = 0.1

    # iterate over the data, with a window length of one hour and 2x10 seconds of
    # overlap
    for traces in p.chopper(tmin=tmin, tinc=tinc, tpad=tpad):

        if traces: # the list could be empty due to gaps
            for tr in traces:
                tr.downsample_to(target_deltat, snap=True, demean=False)

                # remove overlapping
                tr.chop(tr.wmin, tr.wmax)

            window_start = traces[0].wmin
            timestring = util.time_to_str(window_start, format='%Y-%m-%d_%H')
            filepath = 'downsampled/%(station)s_%(channel)s_%(mytimestring)s.mseed'
            io.save(traces, filepath, additional={'mytimestring': timestring})


    # now look at the result with
    #   > snuffler downsampled/


Converting a dataset from Mini-SEED to SAC format
-------------------------------------------------

Conversion of a mseed file to SAC. See :mod:`pyrocko.io` for supported formats.

::

    from pyrocko import pile, io, util, model
    
    input_path = 'data/mseed'
    output_path = 'data/sac/%(dirhz)s/%(station)s/%(station)s_%(channel)s_%(tmin)s.sac'
    fn_stations = 'meta/stations.txt'
    
    stations_list = model.load_stations(fn_stations)
    
    stations = {}
    for s in stations_list:
        stations[s.network, s.station, s.location] = s
        s.set_channels_by_name(*'BHN BHE BHZ BLN BLE BLZ'.split())

    p = pile.make_pile(input_path)
    h = 3600.
    tinc = 1*h
    tmin = util.day_start(p.tmin)
    for traces in p.chopper_grouped(tmin=tmin, tinc=tinc, gather=lambda tr: tr.nslc_id):
        for tr in traces:
            dirhz = '%ihz' % int(round(1./tr.deltat))
            io.save(
                [tr], output_path, 
                format='sac',
                additional={'dirhz': dirhz},
                stations=stations)
