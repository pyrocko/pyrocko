Programming Examples
====================

Load, filter, save
------------------

Read a test file `test.mseed <_static/test.mseed>`_, containing a three component seismogram, apply Butterworth lowpass filter to the seismograms and dump the results to a new file.

::

    from pyrocko import io

    traces = io.load('test.mseed')
   
    for tr in traces:
        tr.lowpass(4, 0.02)   # 4th order, 0.02 Hz
    
    io.save(traces, 'filtered.mseed')

Quickly look at a trace
-----------------------

To visualize a single trace, use the :py:meth:`pyrocko.trace.Trace.snuffle` method. To look at a list of traces, use the :py:func:`pyrocko.trace.snuffle` function. If you want to see the contents of a pile, the :py:meth:`pyrocko.pile.Pile.snuffle` method is your friend. Alternatively, you could of course save the traces to file and use the standalone Snuffler to look at them.

::
     
    from pyrocko import io, trace, pile

    traces = io.load('test.mseed')
    traces[0].snuffle() # look at a single trace
    trace.snuffle(traces) # look at a bunch of traces

    # do something with the traces:
    new_traces = []
    for tr in traces:
        new = tr.copy()
        new.whiten()
        # to allow the viewer to distinguish the traces
        new.set_location('whitened') 
        new_traces.append(new)

    trace.snuffle(traces + new_traces)

    # it is also possible to 'snuffle' a pile:
    p = pile.make_pile(['test.mseed'])
    p.snuffle()


Create a trace object from scratch
----------------------------------

::

    from pyrocko import trace, util, io
    import numpy as num

    nsamples = 100
    tmin = util.str_to_time('2010-02-20 15:15:30.100')
    data = num.random.random(nsamples)
    t1 = trace.Trace(station='TEST', channel='Z', deltat=0.5, tmin=tmin, ydata=data)
    t2 = trace.Trace(station='TEST', channel='N', deltat=0.5, tmin=tmin, ydata=data)
    io.save([t1,t2], 'my_precious_traces.mseed')            # all traces in one file
    io.save([t1,t2], 'my_precious_trace_%(channel)s.mseed') # each file one channel

Extracting part of a trace
----------------------------------

::

    from pyrocko import io
    
    traces = list(io.load('test.mseed'))
    t = traces[0]
    print 'original:', t
    
    # extract a copy of a part of t
    extracted = t.chop(t.tmin+10, t.tmax-10, inplace=False)
    print 'extracted:', extracted
    
    # in-place operation modifies t itself
    t.chop(t.tmin+10, t.tmax-10)
    print 'modified:', t

Reorganizing a dataset into hour-files
----------------------------------------


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

* in each iteration we get all data for the current time window as a list of traces
* the traces emitted by :py:meth:`pyrocko.pile.Pile.chopper()` 'know' the time window to which
  they belong; it is stored in the attributes ``trace.wmin`` and ``trace.wmax``.
  note: ``trace.tmin`` (its onset) does not have to be identical to ``trace.wmin``
* directory parts in the output path will be created as neccessary
* when applying this procedure to a dataset consisting of arbitrarily separated files, it will automatically connect adjacent traces as needed!

Downsampling a whole dataset
----------------------------------

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

Convert SAC to MiniSEED
---------------------------

A very basic SAC to MiniSEED converter::

    from pyrocko import io
    import sys

    for filename in sys.argv[1:]:
        traces = io.load(filename, format='sac')
        if filename.lower().endswith('.sac'):
            out_filename = filename[:-4] + '.mseed'
        else:
            out_filename = filename + '.mseed'

        io.save(traces, out_filename)


Convert MiniSEED to ASCII
-----------------------------

An inefficient, non-portable, non-header-preserving, but simple, method to convert some MiniSEED traces to ASCII tables::

    from pyrocko import io
    
    traces = io.load('test.mseed')
    
    for it, t in enumerate(traces):
        f = open('test-%i.txt' % it, 'w')
        
        for tim, val in zip(t.get_xdata(), t.get_ydata()):
            f.write( '%20f %20g\n' % (tim,val) )
        
        f.close()

Restitute traces to displacement using poles and zeros
----------------------------------------------------------

Often we want to deconvolve instrument responses from seismograms. The method
:py:meth:`pyrocko.trace.Trace.transfer` implements a convolution with a
transfer function in the frequency domain. This method takes as argument a
transfer function object which 'knows' how to compute values of the transfer
function at given frequencies. The trace module provides a few different
transfer functions, but it is also possible to write a custom transfer
function. For a transfer function given as poles and zeros, we can use
instances of the class :py:class:`pyrocko.trace.PoleZeroResponse`. There is
also a class :py:class:`InverseEvalrespResponse`, which uses the common ``RESP`` files
through the ``evalresp`` library.

Here is a complete example using a SAC pole-zero file
(`STS2-Generic.polezero.txt <_static/STS2-Generic.polezero.txt>`_) to
deconvolve the transfer function from an example seismogram::

    from pyrocko import pz, io, trace
    
    # read poles and zeros from SAC format pole-zero file
    zeros, poles, constant = pz.read_sac_zpk('STS2-Generic.polezero.txt')
    
    zeros.append(0.0j)  # one more for displacement
    
    # create pole-zero response function object for restitution, so poles and zeros
    # from the response file are swapped here.
    rest_sts2 = trace.PoleZeroResponse(poles, zeros, 1./constant)
    
    traces = io.load('test.mseed')
    out_traces = []
    for trace in traces:
        
        displacement =  trace.transfer(
            1000.,                       # rise and fall of time domain taper in [s]
            (0.001, 0.002, 5., 10.),     # frequency domain taper in [Hz]
            transfer_function=rest_sts2)
        
        # change channel id, so we can distinguish the traces in a trace viewer.
        displacement.set_codes(channel='D'+trace.channel[-1])
        
        out_traces.append(displacement)
            
    io.save(out_traces, 'displacement.mseed')


Distance between two points
-------------------------------

::

    from pyrocko import orthodrome, model

    e = model.Event(lat=10., lon=20.)
    s = model.Station(lat=15., lon=120.)

    # one possibility:
    d = orthodrome.distance_accurate50m(e,s)
    print 'Distance between e and s is %g km' % (d/1000.)

    # another possibility:
    s.set_event_relative_data(e)
    print 'Distance between e and s is %g km' % (s.dist_m/1000.)

Convert a dataset from Mini-SEED to SAC format
--------------------------------------------------

::

    from pyrocko import pile, io, util, model
    
    dinput = 'data/mseed'
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

Misfit of one trace against two other traces
------------------------------------------------

Three traces will be created. One of these traces will be assumed to be the reference trace (rt) that we want to know the misfit of in comparison to two other traces (tt1 and tt2). The traces rt and tt1 will be provided with the same random y-data. Hence, their misfit will be zero, in the end.

::

    from pyrocko import trace
    from math import sqrt
    import numpy as num
    
    # Let's create three traces: One trace as the reference (rt) and two as test 
    # traces (tt1 and tt2):
    ydata1 = num.random.random(1000)
    ydata2 = num.random.random(1000)
    rt = trace.Trace(station='REF', ydata=ydata1)
    candidate1 = trace.Trace(station='TT1', ydata=ydata1)
    candidate2 = trace.Trace(station='TT2', ydata=ydata2)
    
    # Define a fader to apply before fft.
    taper = trace.CosFader(xfade=5)
    
    # Define a frequency response to apply before performing the inverse fft.
    # This can be basically any funtion, as long as it contains a function called
    # *evaluate*, which evaluates the frequency response function at a given list
    # of frequencies.
    # Please refer to the :py:class:`FrequencyResponse` class or its subclasses for
    # examples.
    # However, we are going to use a butterworth low-pass filter in this example.
    bw_filter = trace.ButterworthResponse(corner=2,
                                          order=4,
                                          typ='low')
    
    # Combine all information in one misfit setup:
    setup = trace.MisfitSetup(description='An Example Setup',
                              norm=2,
                              taper=taper,
                              filter=bw_filter,
                              domain='time_domain')
    
    # Calculate misfits of each candidate against the reference trace:
    for candidate in [candidate1, candidate2]:
        misfit = rt.misfit(candidate=candidate, setup=setup)
        print 'misfit: %s, normalization: %s' % misfit
    
    # Finally, dump the misfit setup that has been used as a yaml file for later
    # re-use:
    setup.dump(filename='my_misfit_setup.txt')
    
If we wanted to reload our misfit setup, guts provides the iload_all() method for 
that purpose:

::

    from pyrocko.guts import load
    from pyrocko.trace import MisfitSetup 
    
    setup = load(filename='my_misfit_setup.txt')
    
    # now, we can change for example only the domain:
    setup.domain = 'frequency_domain'
    
    print setup
