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

