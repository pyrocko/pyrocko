

# this part of the trace.py module needs coroutines, which were introduced with Python2.5

import math
from pyrocko import util
import numpy as num
from scipy import signal

def near(a,b,eps):
    return abs(a-b) < eps

def coroutine(func):
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        gen.next()
        return gen

    wrapper.__name__ = func.__name__
    wrapper.__dict__ = func.__dict__
    wrapper.__doc__ = func.__doc__
    return wrapper

class States:
    '''Utility to store channel-specific state in coroutines.'''
    
    def __init__(self):
        self._states = {}

    def get(self, tr):
        k = tr.nslc_id
        if k in self._states:
            tmin, deltat, value = self._states[k]
            if (near(tmin, tr.tmin, deltat/100.) and
                near(deltat, tr.deltat, deltat/10000.)):
                return value
        
        return None

    def set(self, tr, value):
        self._states[tr.nslc_id] = (tr.tmax+tr.deltat, tr.deltat, value)


@coroutine
def co_list_append(list):
    while True:
        list.append((yield))

@coroutine
def co_lfilter(target, b,a):
    '''Successively filter broken continuous trace data (coroutine).
    
    Create coroutine which takes :py:class:`Trace` objects, filters their data
    through :py:func:`scipy.signal.lfilter` and sends new :py:class:`Trace`
    objects containing the filtered data to target. This is useful, if one
    wants to filter a long continuous time series, which is split into many
    successive traces without producing filter artifacts at trace boundaries.
    
    Filter states are kept *per channel*, specifically, for each (network,
    station, location, channel) combination occuring in the input traces, a
    separate state is created and maintained. This makes it possible to filter
    multichannel or multistation data with only one :py:func:`co_lfilter`
    instance.
    
    Filter state is reset, when gaps occur.
    
    Use it like this::
      
      from pyrocko.trace import co_lfilter, co_list_append

      filtered_traces = []
      pipe = co_lfilter(co_list_append(filtered_traces), a,b)
      for trace in traces:
           pipe.send(trace)

      pipe.close()
    
    '''
    try:
        states = States()
        output = None
        while True:
            input = (yield)
        
            k = input.nslc_id
            zi = states.get(input)
            if zi is None:
                zi = num.zeros(max(len(a), len(b))-1, dtype=num.float)
            
            output = input.copy(data=False)
            ydata, zf = signal.lfilter(b,a, input.get_ydata(),zi=zi)
            output.set_ydata(ydata)
            states.set(input, zf)
            target.send(output)

    except GeneratorExit:
        target.close()

def co_antialias(target, q, n=None, ftype='fir'):
    b,a,n = util.decimate_coeffs(q,n,ftype)
    anti = co_lfilter(target, b,a)
    return anti

@coroutine
def co_dropsamples(target, q, nfir):
    try:
        states = States()
        while True:
            tr = (yield)
            newdeltat = q * tr.deltat
            ioffset = states.get(tr)
            if ioffset is None:
                # for fir filter, the first nfir samples are pulluted by boundary effects; cut it off.
                # for iir this may be (much) more, we do not correct for that.
                # put sample instances to a time which is a multiple of the new sampling interval.
                newtmin_want = math.ceil((tr.tmin+(nfir+1)*tr.deltat)/newdeltat) * newdeltat - (nfir/2*tr.deltat)
                ioffset = int(round((newtmin_want - tr.tmin)/tr.deltat))
                if ioffset < 0:
                    ioffset = ioffset % q

            newtmin_have = tr.tmin + ioffset * tr.deltat
            newtr = tr.copy(data=False)
            newtr.deltat = newdeltat
            newtr.tmin = newtmin_have - (nfir/2*tr.deltat) # because the fir kernel shifts data by nfir/2 samples
            newtr.set_ydata(tr.get_ydata()[ioffset::q].copy())
            states.set(tr, (ioffset % q - tr.data_len() % q ) % q)        
            target.send(newtr)

    except GeneratorExit:
        target.close()

def co_downsample(target, q, n=None, ftype='fir'):
    '''Successively downsample broken continuous trace data (coroutine).

    Create coroutine which takes :py:class:`Trace` objects, downsamples their
    data and sends new :py:class:`Trace` objects containing the downsampled
    data to target.  This is useful, if one wants to downsample a long
    continuous time series, which is split into many successive traces without
    producing filter artifacts and gaps at trace boundaries.
    
    Filter states are kept *per channel*, specifically, for each (network,
    station, location, channel) combination occuring in the input traces, a
    separate state is created and maintained. This makes it possible to filter
    multichannel or multistation data with only one :py:func:`co_lfilter`
    instance.
    
    Filter state is reset, when gaps occur. The sampling instances are choosen
    so that they occur at (or as close as possible) to even multiples of the
    sampling interval of the downsampled trace (based on system time).'''
    b,a,n = util.decimate_coeffs(q,n,ftype)
    return co_antialias(co_dropsamples(target,q,n), q,n,ftype)
        
@coroutine
def co_downsample_to(target, deltat):

    decimators = {}
    try:
        while True:
            tr = (yield)
            ratio = deltat / tr.deltat
            rratio = round(ratio)
            if abs(rratio - ratio)/ratio > 0.0001:
                raise util.UnavailableDecimation('ratio = %g' % ratio)
            
            deci_seq = tuple( x for x in util.decitab(int(rratio)) if x != 1 )
            if deci_seq not in decimators:
                pipe = target
                for q in deci_seq[::-1]:
                    pipe = co_downsample(pipe, q)

                decimators[deci_seq] = pipe
               
            decimators[deci_seq].send( tr )

    except GeneratorExit:
        for g in decimators.values():
            g.close()

