from pyrocko import pz, io, trace
from pyrocko.example import get_example_data

# Download example data
get_example_data('STS2-Generic.polezero.txt')
get_example_data('test.mseed')

# read poles and zeros from SAC format pole-zero file
zeros, poles, constant = pz.read_sac_zpk('STS2-Generic.polezero.txt')

# one more zero to convert from velocity->counts to displacement->counts
zeros.append(0.0j)

rest_sts2 = trace.PoleZeroResponse(
    zeros=zeros,
    poles=poles,
    constant=constant)

traces = io.load('test.mseed')
out_traces = list(traces)
for tr in traces:

    displacement = tr.transfer(
        1000.,                    # rise and fall of time window taper in [s]
        (0.001, 0.002, 5., 10.),  # frequency domain taper in [Hz]
        transfer_function=rest_sts2,
        invert=True)              # to change to (counts->displacement)

    # change channel id, so we can distinguish the traces in a trace viewer.
    displacement.set_codes(channel='D'+tr.channel[-1])

    out_traces.append(displacement)

io.save(out_traces, 'displacement.mseed')
