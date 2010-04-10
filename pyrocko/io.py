
import os
import mseed, sac, kan, segy
import trace
from pyrocko.mseed_ext import MSeedError

class FileLoadError(Exception):
    pass

def make_substitutions(tr, substitutions):
    if substitutions:
        for k,v in substitutions.iteritems():
            if hasattr(tr, k):
                setattr(tr, k, v)

def load(filename, format='mseed', getdata=True, substitutions=None ):
    '''Load traces from file.
    
    Inputs:
        format -- format of the file ('mseed', 'sac', 'kan', 'from_extension', 'try')
        substitutions -- dict with substitutions to be applied to the traces
           metadata
    
    Outputs:
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
    
    
def save(traces, filename_template, format='mseed'):
    if format == 'mseed':
        return mseed.save(traces, filename_template)
    
    elif format == 'sac':
        fns = []
        for tr in traces:
            f = sac.SacFile(from_trace=tr)
            fn = tr.fill_template(filename_template)
            f.write(fn)
            fns.append(fn)
            
        return fns
        