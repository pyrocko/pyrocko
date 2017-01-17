
import random
import time
import tempfile
import os
import fcntl
import unittest
import errno
from pyrocko import util
from pyrocko.parimap import parimap


class Crash(Exception):
    pass


def imapemulation(function, *iterables):
    iterables = map(iter, iterables)
    while True:
        args = [next(it) for it in iterables]
        if function is None:
            yield tuple(args)
        else:
            yield function(*args)


class ParimapTestCase(unittest.TestCase):

    def test_parimap(self):

        # random.seed(0)

        for i in range(50):
            nprocs = random.randint(1, 10)
            nx = random.randint(0, 1000)
            ny = random.randint(0, 1000)
            icrash = random.randint(0, 1000)

            # print 'testing %i %i %i %i...' % (nprocs, nx, ny, icrash)

            def work(x, y):
                if x == icrash:
                    raise Crash(str((x, y)))

                a = random.random()
                if a > 0.5:
                    time.sleep((a*0.01)**2)

                return x+y

            I1 = parimap(
                work, xrange(nx), xrange(ny),
                nprocs=nprocs, eprintignore=Crash)

            I2 = imapemulation(work, xrange(nx), xrange(ny))

            while True:
                e1, e2 = None, None
                r1, r2 = None, None
                end1, end2 = None, None
                try:
                    r1 = I1.next()
                except StopIteration:
                    end1 = True
                except Crash, e1:
                    pass

                try:
                    r2 = I2.next()
                except StopIteration:
                    end2 = True
                except Crash, e2:
                    pass

                assert r1 == r2, str((r1, r2))
                assert type(e1) == type(e2)
                assert end1 == end2

                if end1 or end2:
                    break

    def test_locks(self):

        def work(x):
            assert os.path.exists(fn)
            f = open(fn, 'a+')
            while True:
                try:
                    fcntl.lockf(f, fcntl.LOCK_EX)
                    break
                except IOError, e:
                    if e.errno == errno.ENOLCK:
                        time.sleep(0.01)
                        pass
                    else:
                        raise

            f.seek(0)
            assert '' == f.read()
            f.write('%s' % x)
            f.flush()
            # time.sleep(0.01)
            f.seek(0)
            f.truncate(0)
            fcntl.lockf(f, fcntl.LOCK_UN)
            f.close()

        fos, fn = tempfile.mkstemp()  # (dir='/try/with/nfs/mounted/dir')
        f = open(fn, 'w')
        f.close()

        for x in parimap(work, xrange(100), nprocs=10, eprintignore=()):
            pass

        os.close(fos)
        os.remove(fn)


if __name__ == '__main__':
    util.setup_logging('test_parimap', 'warning')
    unittest.main()
