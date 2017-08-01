Seismic traces
==============

Pyrocko can be used to handle and process seismic waveform data. The following
examples describe usage of the :mod:`pyrocko.io` module to read and write
seismological waveform data and the :mod:`pyrocko.trace` module which offers
basic signal processing functionality.

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
GSE2         gse2                        yes       yes
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
---------------------

Read a test file :download:`test.mseed </static/test.mseed>` with
:func:`pyrocko.io.load`, containing a three component seismogram, apply
Butterworth lowpass filter to the seismograms and dump the results to a new
file with :meth:`pyrocko.io.save`.

::

    from pyrocko import io

    traces = io.load('test.mseed')

    for tr in traces:
        tr.lowpass(4, 0.02)   # 4th order, 0.02 Hz

    io.save(traces, 'filtered.mseed')

Other filtering methods are :meth:`pyrocko.trace.Trace.highpass` and
:meth:`pyrocko.trace.Trace.bandpass`.

If more than a single file should be read, it is much more convenient to use
Pyrocko's :mod:`pyrocko.pile` module instead of :func:`pyrocko.io.load`. See
section :doc:`/library/examples/dataset_management` for examples on how to use
it.

Visual inspection of traces
---------------------------

To visualize a single :class:`~pyrocko.trace.Trace` object, use its
:meth:`~pyrocko.trace.Trace.snuffle` method. To look at a list of traces, use
the :func:`pyrocko.trace.snuffle` function. If you want to see the contents of
a pile, the :meth:`pyrocko.pile.Pile.snuffle` method is your friend.
Alternatively, you could of course save the traces to file and use the
standalone :doc:`/apps/snuffler/index` to look at them.

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

If needed, station meta-information, event information, and marker objects can
be passed into any of the ``snuffle()`` methods and  functions using keyword
arguments.


Creating a trace object from scratch
------------------------------------

Creates two :class:`~pyrocko.trace.Trace` objects, fills them with noise (using
:func:`numpy.random.random`) and saves them with :func:`~pyrocko.io.save` to a
single or to split files. For each :class:`~pyrocko.trace.Trace` object the
station name is defined, the channel name, the sampling interval (0.5 s) and
the time onset (``tmin``).

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

Trimming is achieved with the :meth:`~pyrocko.trace.Trace.chop` method. Here we
cut 10 s from the beginning and the end of the example trace
(:download:`test.mseed </static/test.mseed>`).

::

    from pyrocko import io

    traces = io.load('test.mseed')
    tr = traces[0]  # reference first trace as tr
    print('original:', tr)

    # extract a copy of a part of tr
    tr_extracted = tr.chop(tr.tmin+10.0, tr.tmax-10.0, inplace=False)
    print('extracted:', tr_extracted)

    # in-place operation modifies tr itself
    tr.chop(tr.tmin+10.0, tr.tmax-10.0)
    print('modified:', tr)



Time shifting a trace
---------------------

This example demonstrates how to time shift a trace by a given relative time or
to a given absolute onset time with :meth:`pyrocko.trace.Trace.shift`.

::

    from pyrocko import io, util

    traces = io.load('test.mseed')
    tr = traces[0]

    # shift by 10 seconds backward in time
    tr.shift(-10.0)
    print(tr)

    # shift to a new absolute onset time
    tmin_new = util.str_to_time('2009-04-06 01:32:42.000')
    tr.shift(tmin_new - tr.tmin)
    print(tr)


Resampling a trace
------------------

Example for downsampling a trace in a file to a sampling rate with
:meth:`pyrocko.trace.Trace.downsample_to`.

::

    from pyrocko import io, trace

    tr1 = io.load('test.mseed')[0]

    tr2 = tr1.copy()
    tr2.downsample_to(2.0)

    # make them distinguishable
    tr1.set_location('1')
    tr2.set_location('2')

    # visualize with Snuffler
    trace.snuffle([tr1, tr2])

To overlay the traces in Snuffler, right-click the mouse button and

* check '*Subsort ... (Grouped by Location)*'
* uncheck '*Show Boxes*'
* check '*Common Scale*'


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

An inefficient, non-portable, non-header-preserving, but simple, method to
convert some MiniSEED traces to ASCII tables::

    from pyrocko import io

    traces = io.load('test.mseed')

    for it, t in enumerate(traces):
        f = open('test-%i.txt' % it, 'w')

        for tim, val in zip(t.get_xdata(), t.get_ydata()):
            f.write( '%20f %20g\n' % (tim,val) )

        f.close()


Finding the comparative misfits of mulitple traces
--------------------------------------------------

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
    taper = trace.CosFader(xfade=5.0)

    # Define a frequency response to apply before performing the inverse fft.
    # This can be basically any funtion, as long as it contains a function called
    # *evaluate*, which evaluates the frequency response function at a given list
    # of frequencies.
    # Please refer to the :class:`FrequencyResponse` class or its subclasses for
    # examples.
    # However, we are going to use a butterworth low-pass filter in this example.
    bw_filter = trace.ButterworthResponse(corner=2.0,
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
        print('misfit: %s, normalization: %s' % misfit)

    # Finally, dump the misfit setup that has been used as a yaml file for later
    # re-use:
    setup.dump(filename='my_misfit_setup.txt')

If we wanted to reload our misfit setup, :mod:`pyrocko.guts` provides the
``iload_all()`` method for that purpose:

::

    from pyrocko.guts import load
    from pyrocko.trace import MisfitSetup

    setup = load(filename='my_misfit_setup.txt')

    # now we can change, for example, the domain:
    setup.domain = 'frequency_domain'

    print(setup)


Restitute to displacement using poles and zeros
-----------------------------------------------

Often we want to deconvolve instrument responses from seismograms. The method
:meth:`pyrocko.trace.Trace.transfer` implements a convolution with a transfer
function in the frequency domain. This method takes as argument a transfer
function object which 'knows' how to compute values of the transfer function at
given frequencies. The trace module provides a few different transfer
functions, but it is also possible to write a custom transfer function. For a
transfer function given as poles and zeros, we can use instances of the class
:class:`pyrocko.trace.PoleZeroResponse`. There is also a class
:class:`pyrocko.trace.InverseEvalrespResponse`, which uses the common ``RESP``
files through the ``evalresp`` library.

Here is a complete example using a SAC pole-zero file
(:download:`STS2-Generic.polezero.txt </static/STS2-Generic.polezero.txt>`) to
deconvolve the transfer function from an example seismogram:

Download :download:`trace_handling_example_pz.py </../../src/tutorials/trace_handling_example_pz.py>`


.. literalinclude :: /../../src/tutorials/trace_handling_example_pz.py
    :language: python
