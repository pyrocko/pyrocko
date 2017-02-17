Dataset management
==================


Reorganizing a dataset into hour-files
--------------------------------------
In each iteration we get all data for the current time window as a list of traces. The traces emitted by :py:meth:`pyrocko.pile.Pile.chopper()` 'know' the time window to which they belong; it is stored in the attributes ``trace.wmin`` and ``trace.wmax``.
note: ``trace.tmin`` (its onset) does not have to be identical to ``trace.wmin``. The directory parts in the output path will be created as neccessary.
When applying this procedure to a dataset consisting of arbitrarily separated files, it will automatically connect adjacent traces as needed!

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

    from pyrocko import io, pile, util
    
    outfn = 'resampled_corrtime'
    
    infn = 'displacement'
    trace_pile = pile.make_pile(infn)
    
    def resample(tr):
    
        mindt=2.
        print tr.ydata
        try:
            tr.downsample_to(mindt)
            io.save(tr, '%s/DISPL.%s.%s'%(outfn, tr.station, tr.channel))
            print 'SAVED'
        except util.UnavailableDecimation as e:
            print e
            print 'skip ', tr
       
    
    traces = []
    traces_iterator = trace_pile.iter_all(load_data=True)
    for tr in traces_iterator:
        print tr.ydata
        traces.append(tr)
    
    for tr in traces:
        print tr.ydata
    
    for tr in traces:
        resample(tr)
        
        
        
        
        

Shifting and downsampling a whole dataset with parallel processing
-------------------------------------------------------------------
Example for downsampling all trace files in the input folder to a common sampling rate with :py:meth:`pyrocko.trace.Trace.downsample_to` and shifting them all to a common beginning time with 
:py:meth:`~pyrocko.trace.Trace.shift`.

The shifted and resampled traces are saved into a output folder. This is done in a simple parallel processing loop.
::

    
    from pyrocko import io, pile, util
    from multiprocessing import Pool
    
    outfn = 'resampled_corrtime'
    
    infn = 'input' #input folder with single files as mseed or several dataset
    
    mindt=2.  #resampling [s]
    trace_pile = pile.make_pile(infn)
    
    for t in trace_pile.iter_traces():
        mindt = min(mindt, t.deltat)
    
    
    tshift = -1*util.str_to_time('2009-04-06 01:32:42.000')  #shift your onset of traces to this time
    def resample_parallel(t):
        
        print t.ydata
        try:   
            t.shift(tshift)  #shift your trace object t
            t.downsample_to(mindt)
            io.save(t, '%s/DISPL.%s.%s'%(outfn, t.station, t.channel))
            print 'SAVED'
        except util.UnavailableDecimation as e:
            print e
            print 'skip ', t
       
    traces = []
    traces_iterator = trace_pile.iter_all(load_data=True)
    for t in traces_iterator:  #append all the traces in the pile into the t
        print t.ydata
        traces.append(t)
    
    for t in traces:
        print tr.ydata
    p = Pool(4)  #number of cores
    map(resample_parallel, traces)
    for t in traces:
        resample_parallel(t)
    p.terminate() 



Convert a dataset from Mini-SEED to SAC format
--------------------------------------------------
Conversion of mseed to SAC.
::

    from pyrocko import pile, io, util, model
    
    dinput = 'data/mseed'  #input
    doutput = 'data/sac/%(dirhz)s/%(station)s/%(station)s_%(channel)s_%(tmin)s.sac'
    fn_stations = 'meta/stations.txt'
    
    stations_list = model.load_stations(fn_stations)
    
    stations = {}
    for s in stations_list:
        stations[s.network, s.station, s.location] = s
        s.set_channels_by_name(*'BHN BHE BHZ BLN BLE BLZ'.split())

    p = pile.make_pile(dinput, cachedirname='/tmp/snuffle_cache_u254023')
    h = 3600.
    tinc = 1*h
    tmin = util.day_start(p.tmin)
    for traces in p.chopper_grouped(tmin=tmin, tinc=tinc, gather=lambda tr: tr.nslc_id):
        for tr in traces:
            dirhz = '%ihz' % int(round(1./tr.deltat))
            io.save([tr], doutput, format='sac', additional={'dirhz': dirhz}, stations=stations)
