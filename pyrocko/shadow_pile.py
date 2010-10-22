from pyrocko import pile, util, io
import math, os

pjoin = os.path.join

class ShadowBlock:
    def __init__(self):
        self.mtime = None
        self.files = []

class ShadowPile(pile.Pile):

    def __init__(self, basepile, tinc, tpad=0., storedir='shadow-store'):
        pile.Pile.__init__(self)
        self._base = basepile
        self._tinc = tinc
        self._tpad = tpad
        self._blocks = {}
        self._storedir = storedir
    
    def chopper(self, tmin=None, tmax=None, tinc=None, tpad=0., *args, **kwargs):
        
        if tmin is None:
            tmin = self.base.tmin+tpad
                
        if tmax is None:
            tmax = self.base.tmax-tpad
            
        self._update_range(tmin,tmax)
            
        return pile.Pile.chopper(self, tmin, tmax, tinc, tpad, *args, **kwargs)
        
    def process(self, iblock, tmin, tmax, traces):
        return traces
    
    def _update_range(self, tmin, tmax):
        imin = int(math.floor(tmin / self._tinc))
        imax = int(math.floor(tmax / self._tinc)+1)
        
        todo = []
        for i in xrange(imin, imax):
            wmin = i * self._tinc
            wmax = (i+1) * self._tinc
            mtime = util.gmctime(self._base.get_newest_mtime(wmin,wmax))
            if i not in self._blocks or self._blocks[i].mtime != mtime:
                if i not in self._blocks:
                    self._blocks[i] = ShadowBlock()
                    
                todo.append(i)
                self._blocks[i].mtime = mtime
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
            tmin = iblock*self._tinc
            tmax = (iblock+1)*self._tinc
            traces = self.process(iblock, tmin, tmax, traces)
            if self._tpad != 0.0:
                for trace in traces:
                    trace.chop(tmin, tmax, inplace=True)
            self._clearblock(iblock)
            self._insert(iblock, traces)
            iblock += 1
        
    def _insert(self, iblock, traces):
        fns = io.save(traces, pjoin(self._storedir, '%i.mseed' % iblock))
        files = self.add_files(fns, show_progress=False)
        self._blocks[iblock].files.extend(files)
        
    def _clearblock(self, iblock):
        for file in self._blocks[iblock].files:
            self.remove_file(file)
            