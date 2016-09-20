from pyrocko import pz, io, trace

# read poles and zeros from SAC format pole-zero file
zeros, poles, constant = pz.read_sac_zpk('STS2-Generic.polezero.txt')

zeros.append(0.0j)  # one more for displacement

# create pole-zero response function object for restitution, so poles and zeros
# from the response file are swapped here.
rest_sts2 = trace.PoleZeroResponse(poles, zeros, 1./constant)

traces = io.load('test.mseed')
out_traces = []
for tr in traces:

    displacement = tr.transfer(
        1000.,                      # rise and fall of time domain taper in [s]
        (0.001, 0.002, 5., 10.),    # frequency domain taper in [Hz]
        transfer_function=rest_sts2)

    # change channel id, so we can distinguish the traces in a trace viewer.
    displacement.set_codes(channel='D'+tr.channel[-1])

    out_traces.append(displacement)

io.save(out_traces, 'displacement.mseed')
