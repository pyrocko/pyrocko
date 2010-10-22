from pyrocko import pile, util, io
import math, os

pjoin = os.path.join

class ShadowBlock:
    

class ShadowPile(pile.Pile):

    def __init__(self, basepile, tinc, tpad=0., storedir='shadow-store'):
        pile.Pile.__init__(self)
        self._base = basepile
        self._tinc = tinc
        self._tpad = tpad
        self._block_mtimes = {}
        self._storedir = storedir
    
    def chopper(self, tmin=None, tmax=None, tinc=None, tpad=0., *args, **kwargs):
        
        if tmin is None:
            tmin = self.base.tmin+tpad
                
        if tmax is None:
            tmax = self.base.tmax-tpad
            
        self._update_range(tmin,tmax)
            
        return pile.Pile.chopper(self, tmin, tmax, tinc, tpad, *args, **kwargs)
        
    def _update_range(self, tmin, tmax):
        imin = int(math.floor(tmin / self._tinc))
        imax = int(math.floor(tmax / self._tinc)+1)
        
        todo = []
        for i in xrange(imin, imax):
            wmin = i * self._tinc
            wmax = (i+1) * self._tinc
            mtime = util.gmctime(self._base.get_newest_mtime(wmin,wmax))
            if i not in self._block_mtimes or self._block_mtimes[i] != mtime:
                todo.append(i)
                self._block_mtimes[i] = mtime
            else:
                if todo:
                    self._process_blocks(todo[0], todo[-1]+1)
                    todo = []
        if todo:
            self._process_blocks(todo[0], todo[-1]+1)

                
    def _process_blocks(self, imin, imax):
        pmin = imin * self._tinc
        pmax = imax * self._tinc
        
        iblock = imin
        for traces in self._base.chopper(pmin, pmax, self._tinc, self._tpad):
            ptraces = self.process(iblock, traces)
            self._insert(iblock, ptraces)
            iblock += 1
        
    def process(self, iblock, traces):
        return traces

    def _insert(self, iblock, traces):
        for trace in traces:
            print iblock, trace
        
        fns = io.save(traces, pjoin(self._storedir, '%i.mseed' % iblock))
        self.add_files(fns, show_progress=False)


pbase = pile.make_pile()

p = ShadowPile(pbase, 36000., 360)

tmin = util.ctimegm('2009-05-01 02:30:00')
tmax = util.ctimegm('2009-05-05 01:10:00')


for traces in p.chopper( tmin=tmin, tmax=tmax):
    for trace in traces:
        print trace
        
        

