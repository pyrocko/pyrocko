Seismic trace processing
========================


Misfit of one trace against two other traces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    
If we wanted to reload our misfit setup, guts provides the iload_all() method for 
that purpose:

::

    from pyrocko.guts import load
    from pyrocko.trace import MisfitSetup 
    
    setup = load(filename='my_misfit_setup.txt')
    
    # now, we can change for example only the domain:
    setup.domain = 'frequency_domain'
    
    print setup




Restitute traces to displacement using poles and zeros
-----------------------------------------------

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
