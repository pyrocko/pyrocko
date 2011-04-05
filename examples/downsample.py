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
