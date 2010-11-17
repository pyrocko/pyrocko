

from pyrocko import gse, io, util

util.setup_logging('test_gse', 'debug')


for gse in gse.readgse('test.gse'):
    print gse
    tr = gse.waveforms[0].trace()
    io.save([tr], 'aa.mseed')