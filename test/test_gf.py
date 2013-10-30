
import sys, random
import unittest
from tempfile import mkdtemp
import numpy as num

from pyrocko import gf

def numeq(a,b, eps):
    return (num.all(num.asarray(a).shape == num.asarray(b).shape and
            num.abs(num.asarray(a) - num.asarray(b)) < eps))

class GFTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.tempdirs = []

    def __del__(self):
        import shutil

        for d in self.tempdirs:
            shutil.rmtree(d)

    def create(self, deltat=1.0, nrecords=10 ):
        d = mkdtemp(prefix='gfstore')
        store = gf.store.Store_.create( d, deltat, nrecords, force=True)

        store = gf.store.Store_( d, mode='w')
        for i in range(nrecords):
            data = num.asarray(num.random.random(random.randint(0,7)), 
                    dtype=gf.store.gf_dtype)

            tr = gf.store.GFTrace( data=data, itmin=1+i )
            store.put(i,tr)

        store.close()
        self.tempdirs.append(d)
        return d

    def test_get_spans(self):
        nrecords = 8
        random.seed(0)
        num.random.seed(0)

        store = gf.store.Store_(self.create(nrecords=nrecords))
        for deci in (1,2,3,4):
            for i in range(nrecords):
                tr = store.get(i, decimate=deci)
                itmin, itmax = store.get_span(i, decimate=deci)
                self.assertEqual(tr.itmin, itmin)
                self.assertEqual(tr.data.size, itmax-itmin + 1)

        store.close()
        

    def test_partial_get(self):
        nrecords = 8
        random.seed(0)
        num.random.seed(0)

        store = gf.store.Store_(self.create(nrecords=nrecords))
        for deci in (1,2,3,4):
            for i in range(0,nrecords):
                tr = store.get(i, decimate=deci)
                itmin_gf, nsamples_gf = tr.itmin, tr.data.size
                for itmin in range(itmin_gf - nsamples_gf, itmin_gf + nsamples_gf+1):
                    for nsamples in range(0, nsamples_gf*3):
                        tr2 = store.get(i, itmin, nsamples, decimate=deci)
                        self.assertEqual( tr2.itmin, max(tr.itmin, itmin) )
                        self.assertEqual( tr2.itmin + tr2.data.size, max(min(tr.itmin + tr.data.size, itmin + nsamples), tr2.itmin + tr2.data.size) )
                        
                        ilo = max(tr.itmin, tr2.itmin)
                        ihi = min(tr.itmin+tr.data.size, tr2.itmin+tr2.data.size)

                        a = tr.data[ilo-tr.itmin:ihi-tr.itmin]
                        b = tr2.data[ilo-tr2.itmin:ihi-tr2.itmin]

                        self.assertTrue(numeq(a,b,0.001))

        store.close()


    def test_sum(self):
        
        nrecords = 8 
        random.seed(0)
        num.random.seed(0)

        store = gf.store.Store_(self.create(nrecords=nrecords))

        for deci in (1,2,3,4):
            for i in range(300):
                n = random.randint(0,5)
                indices = num.random.randint(nrecords, size=n)
                weights = num.random.random(n)
                shifts = num.random.random(n)*nrecords
                shifts[::2] = num.round(shifts[::2])

                for itmin,nsamples in [
                        (None, None), 
                        (random.randint(0, nrecords), random.randint(0, nrecords)) ]:

                        a = store.sum(indices, shifts, weights, itmin=itmin, nsamples=nsamples, decimate=deci)
                        b = store.sum_reference(indices, shifts, weights, itmin=itmin, nsamples=nsamples, decimate=deci)

                        self.assertEqual(a.itmin, b.itmin)
                        self.assertTrue(numeq(a.data,b.data, 0.01))



        store.close()


if __name__ == '__main__':
    util.setup_logging('test_gf', 'warning')
    unittest.main()

