'''
Input and output of seismic traces.

This module provides a simple unified interface to load and save traces to a
few different file formats.  The data model used for the
:py:class:`pyrocko.trace.Trace` objects in Pyrocko is most closely matched by
the Mini-SEED file format.  However, a difference is, that Mini-SEED limits the
length of the network, station, location, and channel codes to 2, 5, 2, and 3
characters, respectively.

============ =========================== ========= ======== ======
format       format identifier           load      save     note
============ =========================== ========= ======== ======
Mini-SEED    mseed                       yes       yes      
SAC          sac                         yes       yes      [#f1]_
SEG Y rev1   segy                        some               
SEISAN       seisan, seisan_l, seisan_b  yes                [#f2]_
KAN          kan                         yes                [#f3]_
YAFF         yaff                        yes       yes      [#f4]_
ASCII Table  text                                  yes      [#f5]_
============ =========================== ========= ======== ======

.. rubric:: Notes

.. [#f1] For SAC files, the endianness is guessed. Additional header information is stored in the :py:class:`Trace`'s ``meta`` attribute. 
.. [#f2] Seisan waveform files can be in little (``seisan_l``) or big endian (``seisan_b``) format. ``seisan`` currently is an alias for ``seisan_l``.
.. [#f3] The KAN file format has only been seen once by the author, and support for it may be removed again.
.. [#f4] YAFF is an in-house, experimental file format, which should not be released into the wild.
.. [#f5] ASCII tables with two columns (time and amplitude) are output - meta information will be lost.
'''

import os
import mseed, sac, kan, segy, yaff, file, seisan_waveform, util
import trace
from pyrocko.mseed_ext import MSeedError
import numpy as num

def load(filename, format='mseed', getdata=True, substitutions=None ):
    '''Load traces from file.

    :param format: format of the file (``'mseed'``, ``'sac'``, ``'segy'``, ``'seisan_l'``, ``'seisan_b'``, ``'kan'``, ``'yaff'``, ``'from_extension'``, or ``'try'``)
    :param getdata: if ``True`` (the default), read data, otherwise only read traces metadata
    :param substitutions:  dict with substitutions to be applied to the traces metadata
    
    :returns: list of loaded traces
    
    When *format* is set to ``'try'``, this function tries to read the files in Mini-SEED, SAC, and YAFF format.
    When *format* is set to ``'from_extension'``, the filename extension is used to decide what format should be assumed. The filename extensions
    considered are (matching is case insensitiv): ``'.sac'``, ``'.kan'``, ``'.sgy'``, ``'.segy'``, ``'.yaff'``, everything else is assumed to be in Mini-SEED format.
    
    This function calls :py:func:`iload` and aggregates the loaded traces in a list.
    '''
    
    return list(iload(filename, format=format, getdata=getdata, substitutions=substitutions))

def iload(filename, format='mseed', getdata=True, substitutions=None ):
    '''Load traces from file (iterator version).
    
    This function works like :py:func:`load`, but returns an iterator which yields the loaded traces.
    '''

    def subs(tr):
        make_substitutions(tr, substitutions)
        return tr
    
    if format == 'from_extension':
        format = 'mseed'
        extension = os.path.splitext(filename)[1]
        if extension.lower() == '.sac':
            format = 'sac'
        elif extension.lower() == '.kan':
            format = 'kan'
        elif extension.lower() in ('.sgy', '.segy'):
            format = 'segy'
        elif extension.lower() == '.yaff':
            format = 'yaff' 

    if format in ('seisan', 'seisan_l', 'seisan_b'):
        endianness = {'seisan_l' : '<', 'seisan_b' : '>', 'seisan' : '<'}[format]
        npad = 4
        try:
            for tr in seisan_waveform.load(filename, load_data=getdata, endianness=endianness, npad=npad):
                yield subs(tr)
        except (OSError, seisan_waveform.SeisanFileError), e:
            raise FileLoadError(e)
    
    if format in ('kan',):
        mtime = os.stat(filename)[8]
        kanf = kan.KanFile(filename, get_data=getdata)
        tr = kanf.to_trace()
        tr.set_mtime(mtime)
        yield subs(tr)
        
        
    if format in ('segy',):
        mtime = os.stat(filename)[8]
        segyf = segy.SEGYFile(filename, get_data=getdata)
        ftrs = segyf.get_traces()
        for tr in ftrs:
            tr.set_mtime(mtime)
            yield subs(tr)
    
    if format in ('yaff', 'try'):
        try:
            for tr in yaff.load(filename, getdata):
                yield subs(tr)
            
        except (OSError, file.FileError), e:
            if format == 'try':
                pass
            else:
                raise FileLoadError(e)
            
    if format in ('sac', 'try'):
        mtime = os.stat(filename)[8]
        try:
            sacf = sac.SacFile(filename, get_data=getdata)
            tr = sacf.to_trace()
            tr.set_mtime(mtime)
            yield subs(tr)
            
        except (OSError,sac.SacError), e:
            if format == 'try':
                pass
            else:
                raise FileLoadError(e)
        
    if format in ('mseed', 'try'):
        try:
            for tr in mseed.load(filename, getdata):
                yield subs(tr)
            
        except (OSError, MSeedError), e:
            raise FileLoadError(e)
    
def save(traces, filename_template, format='mseed', additional={}, stations=None):
    '''Save traces to file(s).
    
    :param traces: a trace or an iterable of traces to store
    :param filename_template: filename template with placeholders for trace
            metadata. Uses normal python '%(placeholder)s' string templates. The following
            placeholders are considered: ``network``, ``station``, ``location``,
            ``channel``, ``tmin`` (time of first sample), ``tmax`` (time of last
            sample), ``tmin_ms``, ``tmax_ms``, ``tmin_us``, ``tmax_us``. The
            versions with '_ms' include milliseconds, the versions with '_us'
            include microseconds.
    :param format: ``mseed``, ``sac``, ``text``, or ``yaff``.
    :param additional: dict with custom template placeholder fillins.
    :returns: list of generated filenames

    .. note:: 
        Network, station, location, and channel codes may be silently truncated
        to file format specific maximum lengthes. 
    '''
    if isinstance(traces, trace.Trace):
        traces = [ traces ]

    if format == 'from_extension':
        format = os.path.splitext(filename_template)[1][1:]

    if format == 'mseed':
        return mseed.save(traces, filename_template, additional)
    
    elif format == 'sac':
        fns = []
        for tr in traces:
            f = sac.SacFile(from_trace=tr)
            if stations:
                s = stations[tr.network, tr.station, tr.location]
                f.stla = s.lat
                f.stlo = s.lon
                f.stel = s.elevation
                f.stdp = s.depth
                f.cmpinc = s.get_channel(tr.channel).dip + 90.
                f.cmpaz = s.get_channel(tr.channel).azimuth

            fn = tr.fill_template(filename_template, **additional)
            util.ensuredirs(fn)
            f.write(fn)
            fns.append(fn)
            
        return fns
   
    elif format == 'text':
        fns = []
        for tr in traces:
            fn = tr.fill_template(filename_template, **additional)
            x,y = tr.get_xdata(), tr.get_ydata()
            num.savetxt(fn, num.transpose((x,y)))
            fns.append(fn)
            
    elif format == 'yaff':
        return yaff.save(traces, filename_template, additional)
    else:
        raise UnknownFormat(format)

class FileLoadError(Exception):
    '''Raised when a problem occurred while loading of a file.'''
    pass

class UnknownFormat(Exception):
    '''Raised when an unsupported format is given.'''
    def __init__(self, ext):
        Exception.__init__(self, 'Unknown file format: %s' % ext)

def make_substitutions(tr, substitutions):
    if substitutions:
        tr.set_codes(**substitutions)

