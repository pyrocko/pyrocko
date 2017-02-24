Seismic traces
===============

Pyrocko brings everything to seismic waveform data conviniently and reliable. The following examples describe the object and syntax of a basic pyrocko feature.

Supported seismological waveform formats are:

============ =========================== ========= ======== ======
format       format identifier           load      save     note
============ =========================== ========= ======== ======
MiniSEED     mseed                       yes       yes
SAC          sac                         yes       yes      [#f1]_
SEG Y rev1   segy                        some
SEISAN       seisan, seisan.l, seisan.b  yes                [#f2]_
KAN          kan                         yes                [#f3]_
YAFF         yaff                        yes       yes      [#f4]_
ASCII Table  text                                  yes      [#f5]_
GSE1         gse1                        some
GSE2         gse2                        some
DATACUBE     datacube                    yes
SUDS         suds                        some
============ =========================== ========= ======== ======

.. rubric:: Notes

.. [#f1] For SAC files, the endianness is guessed. Additional header
    information is stored in the :class:`~pyrocko.trace.Trace`'s ``meta`` attribute.
.. [#f2] Seisan waveform files can be in little (``seisan.l``) or big endian
    (``seisan.b``) format. ``seisan`` currently is an alias for ``seisan.l``.
.. [#f3] The KAN file format has only been seen once by the author, and support
    for it may be removed again.
.. [#f4] YAFF is an in-house, experimental file format, which should not be
    released into the wild.
.. [#f5] ASCII tables with two columns (time and amplitude) are output - meta
    information will be lost.

Load, filter and save
----------------------

Read a test file :download:`test.mseed <../_static/test.mseed>` with :meth:`pyrocko.io.load`, containing a three component seismogram, apply Butterworth lowpass filter to the seismograms and dump the results to a new file with :meth:`pyrocko.io.save`.

::

    from pyrocko import io

    traces = io.load('test.mseed')
   
    for tr in traces:
        tr.lowpass(4, 0.02)   # 4th order, 0.02 Hz
    
    io.save(traces, 'filtered.mseed')

Other filtering methods: :meth:`pyrocko.trace.Trace.highpass` and
:meth:`pyrocko.trace.Trace.bandpass`


Quickly inspect a trace's metadata
----------------------------------

Since the trace object is built on the module :mod:`pyrocko.guts`, to view
metadata (in YAML format) one just needs to use the builtin python ``print``
function.

For specific information about a trace, just inspect the corresponding property,
ie. ``station``.

::

    from pyrocko import io

    traces = io.load('test.mseed')
    t = traces[0]
    print t

    print t.station


Quickly inspect a trace
-----------------------

To visualize a single trace from a file, use the :meth:`pyrocko.trace.Trace.snuffle` method. To look at a list of traces, use the :func:`pyrocko.trace.snuffle` function. If you want to see the contents of a pile, the :meth:`~pyrocko.pile.Pile.snuffle` method is your friend. Alternatively, you could of course save the traces to file and use the standalone :doc:`../apps_snuffler` to look at them.

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

Creates two seismological trace objects with :func:`~pyrocko.trace.Trace` and fill it with noise (:func:`numpy.random.random`) and save it with :func:`~pyrocko.io.save`
in to a single file with different channels for the two traces and one file with both traces in one channel.

For each traceobject the name of the station is defined, the channel, the sampling rate (0.5s) and the onset of the trace is given with tmin.

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

Extracting part of a trace (trimming)
-------------------------------------

Trimming is archived with :func:`pyrocko.io.chop`. Here we cut 10 s from the beginning and the end of the example trace (:download:`test.mseed <../_static/test.mseed>`).

::

    from pyrocko import io
    
    traces = list(io.load('test.mseed'))
    t = traces[0]  #the trace is given to t  
    print 'original:', t
    
    # extract a copy of a part of t
    extracted = t.chop(t.tmin+10, t.tmax-10, inplace=False) # the operation chop is done on the trace t
    print 'extracted:', extracted
    
    # in-place operation modifies t itself
    t.chop(t.tmin+10, t.tmax-10)
    print 'modified:', t
    
    

Time shift a trace
--------------------------
This shifts a trace to a specified time with :meth:`pyrocko.trace.Trace.shift`

::

    from pyrocko import io, util
    traces = list(io.load('test.mseed'))
    t = traces[0]  #the trace is given to t  
    tshift = -1*util.str_to_time('2009-04-06 01:32:42.000')  #shift your onset of traces to this time
    #tshift = -10  #Alternative: shift your onset of trace by -10s
    t.shift(tshift)  #shift your trace object t
    io.save(t, '%s/SHIF.%s.%s'%(outfn, t.station, t.channel)) #save the shifted stations
    print 'SAVED'
    

    
    

Resampling a trace
--------------------------

Example for downsampling a trace in a file to a sampling rate with :meth:`pyrocko.trace.Trace.downsample_to`.

::

    from pyrocko import io, util


    traces = list(io.load('test.mseed'))
    t = traces[0]  #the trace is given to t  
    mindt=2.  #resampling [s]    
    t.downsample_to(mindt)
    io.save(t, '%s/DISPL.%s.%s'%(outfn, t.station, t.channel))
    print 'SAVED'
    

    
    


Convert SAC to MiniSEED
-----------------------

A very basic SAC to MiniSEED converter:

::

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
-------------------------

An inefficient, non-portable, non-header-preserving, but simple, method to convert some MiniSEED traces to ASCII tables::

    from pyrocko import io
    
    traces = io.load('test.mseed')
    
    for it, t in enumerate(traces):
        f = open('test-%i.txt' % it, 'w')
        
        for tim, val in zip(t.get_xdata(), t.get_ydata()):
            f.write( '%20f %20g\n' % (tim,val) )
        
        f.close()


Finding the comparative misfits of mulitple traces
---------------------------------------------

Three traces will be created, where one will be the used as a reference trace
(``rt``).  Using :meth:`pyrocko.trace.Trace.misfit`, we can find the misfits
of the other two traces (``tt1`` and ``tt2``) in comparision to ``rt``.
Traces ``rt`` and ``tt1`` will have the same y-data, so the misfit between
them will be zero.


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
    # Please refer to the :class:`FrequencyResponse` class or its subclasses for
    # examples.
    # However, we are going to use a butterworth low-pass filter in this example.
    bw_filter = trace.ButterworthResponse(corner=2,
                                          order=4,
                                          type='low')
    
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
    
If we wanted to reload our misfit setup, :mod:`pyrocko.guts` provides the ``iload_all()`` method for 
that purpose:

::

    from pyrocko.guts import load
    from pyrocko.trace import MisfitSetup 
    
    setup = load(filename='my_misfit_setup.txt')
    
    # now we can change, for example, the domain:
    setup.domain = 'frequency_domain'
    
    print setup


Restitute to displacement using poles and zeros
--------------------------------------------------

Often we want to deconvolve instrument responses from seismograms. The method
:meth:`pyrocko.trace.Trace.transfer` implements a convolution with a
transfer function in the frequency domain. This method takes as argument a
transfer function object which 'knows' how to compute values of the transfer
function at given frequencies. The trace module provides a few different
transfer functions, but it is also possible to write a custom transfer
function. For a transfer function given as poles and zeros, we can use
instances of the class :class:`pyrocko.trace.PoleZeroResponse`. There is
also a class :class:`pyrocko.trace.InverseEvalrespResponse`, which uses the common ``RESP`` files through the ``evalresp`` library.

Here is a complete example using a SAC pole-zero file (:download:`STS2-Generic.polezero.txt <../_static/STS2-Generic.polezero.txt>`) to deconvolve the transfer function from an example seismogram

::

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
