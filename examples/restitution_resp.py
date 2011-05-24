from pyrocko import pz, io, trace, evalresp_ext
import os


traces = io.load('1989.072.evt.mseed')
out_traces = []
for tr in traces:
        
    
    respfn = tr.fill_template( 'RESP.%(network)s.%(station)s.%(location)s.%(channel)s' )
        
     
    if os.path.exists(respfn):
        
        try:
            resp = trace.InverseEvalresp(respfn, tr, target='dis')
        
            tr.extend(tr.tmin - 100., tr.tmax, fillmethod='repeat')
        
            displacement =  tr.transfer(
                100.,                      # rise and fall of time domain taper in [s]
                (0.005, 0.01, 1., 2.),     # frequency domain taper in [Hz]
                transfer_function=resp)
            
            # change channel id, so we can distinguish the traces in a trace viewer.
            displacement.set_codes(channel='D'+tr.channel[-1])
            
            out_traces.append(displacement)
            
        except evalresp_ext.EvalrespError:
            pass
        
io.save(out_traces, 'displacement.mseed')
