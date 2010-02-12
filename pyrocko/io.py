import mseed, sac
import trace
from pyrocko.mseed_ext import MSEEDERROR

class FileLoadError(Exception):
    pass

def make_substitutions(tr, substitutions):
    if substitutions:
        for k,v in substitutions.iteritems():
            if hasattr(tr, k):
                setattr(tr, k, v)

def load(filename, format='mseed', getdata=True, substitutions=None ):
    trs = []
    if format == 'mseed':
        try:
            for tr in mseed.load(filename, getdata):
                trs.append(tr)
            
        except (OSError, MSEEDERROR), e:
            raise FileLoadError(e)
            
    elif format == 'sac':
        mtime = os.stat(filename)[8]
        try:
            sac = sac.SacFile(filename, get_data=getdata)
            tr = sac.to_trace()
            tr.set_mtime(mtime)
            trs.append(tr)
            
        except (OSError,SacError), e: 
            raise FileLoadError(e)
    
    for tr in trs:
        make_substitutions(tr, substitutions)
        
    return trs
    
    
def save(traces, filename_template, format='mseed'):
    if format == 'mseed':
        mseed.save(traces, filename_template)
    
    elif format == 'sac':
        for tr in traces:
            f = sac.SacFile(from_trace=tr)
            f.write(tr.fill_template(filename_template))
            

            