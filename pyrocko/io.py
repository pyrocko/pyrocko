
import os
import mseed, sac, kan, segy
import trace
from pyrocko.mseed_ext import MSeedError

class FileLoadError(Exception):
    pass

def make_substitutions(tr, substitutions):
    if substitutions:
        tr.set_codes(**substitutions)

def load(filename, format='mseed', getdata=True, substitutions=None ):
    '''Load traces from file.
    
    In:
        format -- format of the file ('mseed', 'sac', 'kan', 'from_extension', 'try')
        substitutions -- dict with substitutions to be applied to the traces
           metadata
    
    Out:
        trs -- list of loaded traces
    '''
    
    if format == 'from_extension':
        format = 'mseed'
        extension = os.path.splitext(filename)[1]
        if extension.lower() == '.sac':
            format = 'sac'
        if extension.lower() == '.kan':
            format = 'kan'
    
    trs = []
    
    if format in ('kan',):
        mtime = os.stat(filename)[8]
        kanf = kan.KanFile(filename, get_data=getdata)
        tr = kanf.to_trace()
        tr.set_mtime(mtime)
        trs.append(tr)
        
    if format in ('segy',):
        mtime = os.stat(filename)[8]
        segyf = segy.SEGYFile(filename, get_data=getdata)
        ftrs = segyf.get_traces()
        for tr in ftrs:
            tr.set_mtime(mtime)
        trs.extend(ftrs)
    
    if format in ('sac', 'try'):
        mtime = os.stat(filename)[8]
        try:
            sacf = sac.SacFile(filename, get_data=getdata)
            tr = sacf.to_trace()
            tr.set_mtime(mtime)
            trs.append(tr)
            
        except (OSError,sac.SacError), e:
            if format == 'try':
                pass
            else:
                raise FileLoadError(e)
        
    if format in ('mseed', 'try'):
        try:
            for tr in mseed.load(filename, getdata):
                trs.append(tr)
            
        except (OSError, MSeedError), e:
            raise FileLoadError(e)
    
    for tr in trs:
        make_substitutions(tr, substitutions)
        
    return trs
    
    
def save(traces, filename_template, format='mseed', additional={}):
    '''Save traces to file(s).
    
    In:
        traces - list of traces to store
        filename_template -- filename template with placeholders for trace
            metadata. Valid placeholders are '%(network)s', '%(station)s', 
            '%(location)s', '%(channel)s', '%(tmin)s', and '%(tmax)s'. Custom
            placeholders can be inserted with 'additional' option below.
        format -- 'mseed' or 'sac'.
        additional -- dict with custom placeholder fillins.
        
    Out:
        List of generated filenames
    '''
    
    if format == 'mseed':
        return mseed.save(traces, filename_template, additional)
    
    elif format == 'sac':
        fns = []
        for tr in traces:
            f = sac.SacFile(from_trace=tr)
            fn = tr.fill_template(filename_template, **additional)
            f.write(fn)
            fns.append(fn)
            
        return fns
        