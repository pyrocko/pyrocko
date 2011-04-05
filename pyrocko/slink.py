import subprocess, re, calendar, time, os, signal, sys, logging
import trace
import numpy as num

logger = logging.getLogger('pyrocko.slink')

class SlowSlink:
    def __init__(self, host='geofon.gfz-potsdam.de', port=18000):
        self.host = host
        self.port = port
        self.running = False
        self.stream_selectors = []
        
    def query_streams(self):
        cmd = [ 'slinktool',  '-Q', self.host+':'+str(self.port) ]
        logger.debug('Running %s' % ' '.join(cmd))
        slink = subprocess.Popen(cmd,stdout=subprocess.PIPE)
        (a,b) = slink.communicate()
        streams = []
        for line in a.splitlines():
            toks = line.split()
            if len(toks) == 9:
                net, sta, loc, cha = toks[0], toks[1], '', toks[2]
            else:
                net, sta, loc, cha = toks[0], toks[1], toks[2], toks[3]
            streams.append((net,sta,loc,cha))
        return streams
    
    def add_stream(self, network, station, location, channel):
        self.stream_selectors.append( '%s_%s:%s.D' % (network, station, channel) )
    
    def add_raw_stream_selector(self, stream_selector):
        self.stream_selectors.append(stream_selector)
    
    def acquisition_start(self):
        
        assert not self.running
        
        streams_sel = ','.join(self.stream_selectors)
        cmd = [ 'slinktool', '-u', '-S', streams_sel, self.host+':'+str(self.port) ]        
        
        logger.debug('Starting %s' % ' '.join(cmd))
        self.slink = subprocess.Popen(cmd,stdout=subprocess.PIPE)
        self.header = None
        self.vals = []
        self.running = True
        
    def acquisition_stop(self):
        if not self.running:
            return
        os.kill( self.slink.pid, signal.SIGTERM)
        logger.debug("Waiting for slinktool to terminate...")
        it = 0
        while self.slink.poll() == -1:
            time.sleep(0.01)
            if it == 200:
                logger.debug("Waiting for slinktool to terminate... trying harder...")
                os.kill( self.slink.pid, signal.SIGKILL)
            
            it += 1
        
        self.running = False
        logger.debug("Done, slinktool has terminated")
        
    def process(self):
        
        line = self.slink.stdout.readline()
        
        if not line:
            return False
        
        toks = line.split(', ')
        if len(toks) != 1:
            nslc = tuple(toks[0].split('_'))
            if len(nslc) == 3: nslc = nslc[0], nslc[1], '', nslc[2]
            nsamples = int(toks[1].split()[0])
            rate = float(toks[2].split()[0])
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
                    self.got_trace(tr)
                    self.vals = []
                    self.header = None
        return True
                        
    def got_trace(self, tr):
        logger.info('Got trace from slinktool: %s' % tr)
        
