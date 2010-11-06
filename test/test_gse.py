from pyrocko import gse, io

for gse in gse.readgse('test.gse'):
    print gse
    tr = gse.waveforms[0].trace()
    io.save([tr], 'aa.mseed')