import subprocess, re, calendar, time
import trace
import numpy as num

class SlowSlink:
    def __init__(self, streams, host, port=18000):
        self.streams = streams
        self.host = host
        self.port = port
        
    def start(self):
        cmd = [ 'slinktool', '-u', '-S', self.streams, self.host+':'+str(self.port) ]
        self.slink = subprocess.Popen(cmd,stdout=subprocess.PIPE)
        self.header = None
        self.vals = []
        self.run()
        
    def run(self):
        while self.process():
            pass
        
    def process(self):
        
        line = self.slink.stdout.readline()
        
        if not line:
            return False
        
        toks = line.split(', ')
        if len(toks) != 1:
            nslc = tuple(toks[0].split('_'))
            nsamples = int(toks[1].split()[0])
            rate = int(toks[2].split()[0])
            st,sms = toks[3].split()[0].split('.')
            us = int(sms)
            tstamp = calendar.timegm(time.strptime(st,'%Y,%j,%H:%M:%S'))+us*0.000001
            if nsamples != 0:
                self.header = nslc, nsamples, rate, tstamp
        else:
            if self.header:
                self.vals.extend([ float(x) for x in line.split() ])
                
                if len(self.vals) == self.header[1]:
                    nslc, nsamples, rate, tstamp = self.header
                    deltat = 1.0/rate
                    net, sta, loc, cha = nslc
                    
                    tr = trace.Trace(network=net, station=sta, location=loc, channel=cha, tmin=tstamp, deltat=deltat, ydata=num.array(self.vals))
                    
                    
                            
                    self.vals = []
                    self.header = None
        return True
                        
    def got_trace(self, tr):
        print 'add', tr
        
if __name__ == '__main__':
    sl = SlowSlink(streams='IU_KONO:BH?.D', host='geofon-cluster.gfz-potsdam.de')
    sl.start()
