# tested with Python 3.5, july-2017 (1.12)
# tested with Python 2.4, september-2007 (1.12)
# tested with Python 2.5, december-2008 (1.12_1)

import sys  # noqa
import unittest
import random
import string
import pickle
from io import BytesIO

from pyrocko import avl


def cmp(a, b):
    return (a > b) - (a < b)


def verify_empty(tree):
    return tree.verify() == 1 and len(tree) == 0


def verify_len(tree, size):
    return tree.verify() == 1 and len(tree) == size


def gen_ints(lo, hi):
    for i in range(lo, hi):
        yield i


# call with gcd(step,modulo)=1
def gen_ints_perm(step, modulo):
    n = random.randint(0, modulo-1)
    for i in range(modulo):
        val = n
        n = (val+step) % modulo
        yield val


def gen_pairs(lo, hi):
    for i in range(lo, hi):
        yield (i, 0)


def gen_iter(a, b):
    return zip(a, b)


def range_tree(lo, hi, compare=None):
    t = avl.new(compare=compare)
    for i in gen_ints(lo, hi):
        t.insert(i)
    return t


def random_int_tree(lo, hi, size=1000, compare=None):
    t = avl.new(compare=compare)
    for i in gen_ints(0, size):
        t.insert(random.randint(lo, hi-1))
    return t


def random_pair_tree(lo, hi, size=1000, compare=None):
    t = avl.new(compare=compare)
    for i in gen_ints(0, size):
        t.insert((random.randint(lo, hi-1), 0))
    return t


# return 1 if tree1 and tree2 hold the same set of keys
def equal_tree(tree1, tree2, compare=cmp):
    if len(tree1) != len(tree2):
        return 0
    i1 = iter(tree1)
    i2 = iter(tree2)
    try:
        while compare(next(i1), next(i2)) == 0:  # noqa
            pass
        return 0
    except StopIteration:
        return 1


# sort tuples by decreasing order of first component
def pair_compare(t, u):
    if t[0] < u[0]:
        return 1
    if t[0] == u[0]:
        return 0
    return -1


# ----
# Test insertions
# ----
class Test_avl_ins(unittest.TestCase):

    def setUp(self):
        self.size = 1000
        self.list = list(range(self.size))
        self._times = 7

    # test avl.new
    def testnew_basic(self):
        for i in range(self._times):
            random.shuffle(self.list)
            t = avl.new(self.list, unique=0)
            self.assertTrue(t.verify() == 1 and len(t) == self.size)
            for n in self.list:
                self.assertTrue(n in t)
        t = None

    # test avl.new unique=0
    def testnew_basic_dupes(self):
        size3 = self.size * 3
        for i in range(self._times):
            random.shuffle(self.list)
            list3 = self.list * 3
            t = avl.new(list3, unique=0)
            self.assertTrue(verify_len(t, size3))
            for n in self.list:
                self.assertTrue(n in t)
        t = None

    # test avl.new unique=1
    def testnew_basic_unique(self):
        for i in range(self._times):
            random.shuffle(self.list)
            list3 = self.list * 3
            t = avl.new(list3, unique=1)
            self.assertTrue(verify_len(t, self.size))
            for n in self.list:
                self.assertTrue(n in t)
        t = None

    # test avl.new with compare function
    def testnew_compare_basic(self):
        random.shuffle(self.list)
        pairs = [(i, 0) for i in self.list]
        for i in range(self._times):
            t = avl.new(pairs, compare=pair_compare)
            self.assertTrue(verify_len(t, self.size))
            for p in pairs:
                self.assertTrue(p in t)

    # test avl.new with compare function, unique=0
    def testnew_compare_dupes(self):
        random.shuffle(self.list)
        pairs = [(i, 0) for i in self.list]
        pairs3 = pairs * 3
        size3 = self.size * 3
        for i in range(self._times):
            t = avl.new(pairs3, compare=pair_compare, unique=0)
            self.assertTrue(verify_len(t, size3))
            for p in pairs:
                self.assertTrue(p in t)

    # test avl.new with compare function, unique=1
    def testnew_compare_unique(self):
        random.shuffle(self.list)
        tuples = [(i, 0) for i in self.list]
        tuples3 = tuples * 3
        for i in range(self._times):
            t = avl.new(tuples3, compare=pair_compare, unique=1)
            self.assertTrue(verify_len(t, self.size))
            for u in tuples:
                self.assertTrue(u in t)

    # comparison failure
    def testins_bad(self):
        t = avl.new([(5, 6), (2, 8), (7, -1), (-1, 1), (2, -3)],
                    compare=pair_compare)
        for i in range(5):
            self.assertRaises(TypeError, t.insert, 66)
            self.assertRaises(TypeError, t.has_key, 66)
            self.assertFalse(66 in t)
            t.insert((3, 3))
            self.assertTrue(t.verify() == 1)

    # test ins and clear
    def testins_big(self):
        big_size = 20000
        for i in range(self._times):
            t = random_int_tree(-big_size, big_size, big_size)
            self.assertTrue(verify_len(t, big_size))
            t.clear()
            self.assertTrue(verify_empty(t))

    def tearDown(self):
        self.list = None


# ----
# Test lookup
# ----
class Test_avl_lookup(unittest.TestCase):

    def setUp(self):
        self.n = 5000
        self.t = range_tree(0, self.n)
        self.bad = list(range(-500, 0)) + list(range(self.n, self.n+500))

    def testlookup_get(self):
        for i in self.bad:
            self.assertRaises(LookupError, self.t.lookup, i)
        for i in gen_ints(0, self.n):
            self.assertTrue(self.t.lookup(i) == i)

    def testlookup_contains(self):
        for i in self.bad:
            self.assertFalse(i in self.t)
        for i in gen_ints(0, self.n):
            self.assertTrue(i in self.t)

    def testlookup_span_one(self):
        for i in gen_ints(0, self.n):
            a, b = self.t.span(i)
            self.assertTrue(b-a == 1)

    def testlookup_span_two(self):
        t = self.t
        for i in gen_ints(0, self.n):
            j = random.randint(i, self.n-1)
            self.assertTrue(t.span(i, j) == (i, j+1))
            self.assertTrue(t.span(j, i) == (i, j+1))
            self.assertTrue(t.span(i, self.n) == (i, self.n))
            self.assertTrue(t.span(i, self.n*2) == (i, self.n))
            self.assertTrue(t.span(-j, i) == (0, i+1))

    def tearDown(self):
        self.t = None
        self.bad = None


# ----
# Test iterations
# ----
class Test_avl_iter(unittest.TestCase):

    def setUp(self):
        self.n = 1000
        self.t = range_tree(0, self.n)
        self.orig = list(range(self.n))

    def testiter_forloop(self):
        list = self.orig[:]
        for i in range(5):
            random.shuffle(list)
            for k in avl.new(list):
                self.assertTrue(k == self.orig[k])

    def testiter_forward(self):
        j = self.t.iter()
        for k in gen_ints(0, self.n):
            self.assertTrue(next(j) == k and j.index() == k and j.cur() == k)
        self.assertRaises(StopIteration, j.__next__)
        self.assertRaises(avl.Error, j.cur)
        self.assertTrue(j.index() == self.n)

    def testiter_backward(self):
        j = self.t.iter(1)
        for k in gen_ints(1, self.n+1):
            self.assertTrue(j.prev()+k == self.n
                            and j.index()+k == self.n
                            and j.cur()+k == self.n)
        self.assertRaises(StopIteration, j.prev)
        self.assertRaises(avl.Error, j.cur)
        self.assertTrue(j.index() == -1)

    def testiter_basic(self):
        t = avl.new()
        j = iter(t)     # before first
        k = t.iter(1)   # after last
        self.assertRaises(StopIteration, j.__next__)
        self.assertRaises(StopIteration, j.prev)
        self.assertRaises(StopIteration, k.__next__)
        self.assertRaises(StopIteration, k.prev)
        t.insert('bb')
        self.assertRaises(StopIteration, j.prev)
        self.assertRaises(StopIteration, k.__next__)
        self.assertTrue(next(j) == 'bb')
        self.assertTrue(k.prev() == 'bb')
        self.assertRaises(StopIteration, j.__next__)
        self.assertRaises(StopIteration, k.prev)
        self.assertTrue(j.prev() == 'bb')
        self.assertTrue(next(k) == 'bb')
        self.assertTrue(j.cur() == 'bb' and k.cur() == 'bb')
        t.insert('aa')
        self.assertTrue(j.prev() == 'aa')
        t.insert('cc')
        self.assertTrue(next(k) == 'cc')

    def testiter_remove(self):
        for start in range(1, self.n+1):
            u = avl.new(self.t)
            self.assertTrue(u.verify() == 1)
            j = iter(u)
            for i in range(start):
                next(j)
            index = j.index()
            self.assertTrue(index == start-1)
            while index < len(u):
                j.remove()
                # self.assertTrue(u.verify() == 1)
                self.assertTrue(j.index() == index)
            self.assertRaises(avl.Error, j.remove)
            self.assertTrue(u.verify() == 1)

    def tearDown(self):
        self.t.clear()
        self.t = None
        self.orig = None


# ----
# Test duplication
# ----
class Test_avl_dup(unittest.TestCase):

    def setUp(self):
        self.n = 5000

    def testdup_basic(self):
        for i in range(10):
            t = random_int_tree(0, self.n, size=self.n)
            u = avl.new(t)
            v = t[:]
            self.assertTrue(u.verify() == 1)
            self.assertTrue(equal_tree(t, u))
            self.assertTrue(v.verify() == 1)
            self.assertTrue(equal_tree(t, v))
        t = avl.new()
        u = avl.new(t)
        self.assertTrue(verify_empty(u))

    def testdup_compare(self):
        for i in range(5):
            t = random_pair_tree(0, self.n, size=self.n, compare=pair_compare)
            u = avl.new(t)
            v = t[:]
            self.assertTrue(u.verify() == 1)
            self.assertTrue(equal_tree(t, u, pair_compare))
            self.assertTrue(v.verify() == 1)
            self.assertTrue(equal_tree(t, v, pair_compare))
        t = avl.new(compare=pair_compare)
        u = avl.new(t)
        self.assertTrue(verify_empty(u))


# ----
# Test deletions
# ----
class Test_avl_del(unittest.TestCase):

    def setUp(self):
        pass

    # empty tree with unique keys
    def testdel_basic(self):
        t = avl.new()
        t.remove(1)
        t.remove(-2)
        self.assertTrue(verify_empty(t))
        t = range_tree(-2000, +2000)
        self.assertTrue(t.verify() == 1)
        n = len(t)
        # no-op
        others = list(range(-2100, -2000)) + list(range(2000, 2100))
        random.shuffle(others)
        for i in others:
            t.remove(i)
        self.assertTrue(verify_len(t, n))
        others = None
        # empty trees
        lst = list(range(-2000, 2000))
        for i in range(10):
            random.shuffle(lst)
            u = avl.new(t)
            for k in lst:
                u.remove(k)
                self.assertFalse(k in u)
            self.assertTrue(verify_empty(u))

    # empty tree with duplicates
    def testdel_lessbasic(self):
        n = 1000
        t = avl.new()
        for i in range(3):
            for k in gen_ints(0, n):
                t.insert(k)
        self.assertTrue(t.verify() == 1)
        # params for sequence
        modulo = n
        step = 37
        for i in range(10):
            u = avl.new(t)
            for k in gen_ints_perm(step, modulo):
                while k in u:
                    u.remove(k)
                self.assertFalse(k in u)
            self.assertTrue(u.verify() == 1)
            self.assertTrue(len(u) == 0, 'len='+str(len(u))+' '+str(u))

    def testdel_bigunique(self):
        t = range_tree(-10000, 10000)
        for i in gen_ints_perm(10007, 20000):
            j = i - 10000
            self.assertTrue(j in t)
            t.remove(j)
            self.assertFalse(j in t)

    def testdel_one(self):
        n = 4000
        t = random_int_tree(0, n, n)
        for i in gen_ints_perm(1993, n):
            e = t[i]
            a1, a2 = t.span(e)
            t.remove(e)
            self.assertTrue(t.verify() == 1)
            b1, b2 = t.span(e)
            self.assertTrue(a2-a1-1 == b2-b1)
            t.insert(e)


# ----
# Test sequence support
# ----
class Test_avl_sequence(unittest.TestCase):

    def geti(self, t, i):
        return t[i]

    def testseq_basic(self):
        t = avl.new()
        for step, modulo in [(491, 2000), (313, 10000)]:
            for i in gen_ints_perm(step, modulo):
                t.insert(i)

            for i in gen_ints(0, modulo):
                self.assertTrue(t.index(i) == i)
                self.assertTrue(t[i] == i)
                self.assertTrue(t[-i-1] == t[modulo-i-1])

            for i in gen_ints(-100, 0):
                self.assertTrue(t.index(i) == -1)
            for i in gen_ints(modulo, modulo+100):
                self.assertTrue(t.index(i) == -1)
            t.clear()
        self.assertRaises(IndexError, self.geti, t, 11)
        self.assertRaises(IndexError, self.geti, t, -5)

    def testseq_index(self):
        t = avl.new()
        n = 5000
        for r in range(3):
            for i in range(3):
                for k in gen_ints_perm(11, n):
                    t.insert(k)
            self.assertTrue(verify_len(t, n*3))
            for i in gen_ints(0, n):
                self.assertTrue(t.index(i) == i*3)
                a, b = t.span(i)
                self.assertTrue((a == i*3) and (b-a == 3))
            t.clear()

    def testseq_insert(self):
        n = 5000
        t = avl.new()
        self.assertRaises(IndexError, t.insert, 'out', 1)
        self.assertRaises(IndexError, t.insert, 'out', -13)
        for i in gen_ints_perm(373, n):
            t.insert(2*i)
        self.assertTrue(t.verify() == 1)
        for i in range(3):
            u = avl.new(t)
            for k in gen_ints(0, n):
                odd = 2*k+1
                u.insert(odd, odd)
                self.assertTrue(u[odd] == odd)
        self.assertTrue(u.verify() == 1)
        self.assertTrue(len(u) == 2*n)
        for i in range(100):
            u.append(2*n)
        for i in range(100):
            u.insert(0, 0)
        self.assertTrue(u.verify() == 1)
        list = None  # noqa

    def testseq_remove(self):
        n = 5000
        t = range_tree(0, n)
        irem = -n-random.randint(1, 100)
        self.assertRaises(IndexError, t.remove_at, irem)
        for i in gen_ints_perm(29, n):
            t.remove_at(i)
            self.assertFalse(i in t)
            t.insert(i, i)
        self.assertTrue(verify_len(t, n))
        # empty the tree
        for i in gen_ints(0, n):
            k = random.randint(0, len(t)-1)
            e = t[k]
            t.remove_at(k)
            self.assertFalse(e in t or t.index(e) != -1)
        self.assertTrue(verify_empty(t))
        self.assertRaises(IndexError, t.remove_at, 0)

    def testseq_spaninsert(self):
        n = 5000
        t = avl.new(compare=lambda x, y: cmp(y, x))
        for i in range(2):
            for i in gen_ints(0, 3*n):
                e = random.randint(-n, n)
                a1, a2 = t.span(e)
                t.insert(e, a2)
                self.assertTrue(t.span(e) == (a1, a2+1))
            self.assertTrue(verify_len(t, 3*n))
            t.clear()

    def testseq_removedupes(self):
        n = 2500
        steps = [501, 7, 1003, 1863]
        repeats = len(steps)
        t = avl.new()
        for step in steps:
            for k in gen_ints_perm(step, n):
                t.insert(k)
        for k in gen_ints_perm(1209, n):
            a, b = t.span(k)
            self.assertTrue(b-a == repeats)
            self.assertTrue(a == 0 or t[a-1] < k)
            self.assertTrue(b == len(t) or t[b] > k)
            for i in range(repeats):
                t.remove_at(a)
            self.assertTrue(t.span(k) == (a, a))
        self.assertTrue(verify_empty(t))

    def testseq_slicedup(self):
        n = 3000
        for i in range(3):
            t = random_int_tree(0, n, 4*n)
            self.assertTrue(equal_tree(t[:], t))

    def testseq_sliceempty(self):
        t = random_int_tree(0, 500, size=1000)
        lim = len(t)
        for i in range(100):
            a = random.randint(0, lim)
            b = random.randint(0, a)
            self.assertTrue(verify_empty(t[a:b]))
            a = random.randint(1, lim)
            b = random.randint(1, a)
            self.assertTrue(verify_empty(t[-b:-a]))

    def testseq_slice(self):
        n = 1000
        t = range_tree(0, n)
        for a in range(n):
            u = t[:a]
            self.assertTrue(verify_len(u, a))
            self.assertTrue(equal_tree(u, range_tree(0, a)))
            u = t[a:]
            self.assertTrue(equal_tree(u, range_tree(a, n)))
        t = None

    # test a+b
    def testseq_sliceconcat(self):
        n = 2000
        e = avl.new()
        t = range_tree(0, n)
        self.assertTrue(verify_empty(e+e))
        self.assertTrue(equal_tree(t+e, t))
        self.assertTrue(equal_tree(e+t, t))
        for a in gen_ints_perm(50, n):
            u = t[:a] + t[a:]
            self.assertTrue(u.verify() == 1)
            self.assertTrue(equal_tree(t, u))
            u = None

    def testseq_concatinplace(self):
        self.assertRaises(TypeError, avl.new().concat, [])
        n = 2000
        e = avl.new()
        t = range_tree(0, n)
        u = t[:]
        u.concat(e)
        self.assertTrue(equal_tree(t, u))
        e.concat(u)
        self.assertTrue(equal_tree(e, u))
        u += avl.new()
        self.assertTrue(equal_tree(t, u))
        e.clear()
        e += t
        self.assertTrue(equal_tree(e, t))
        e.clear()
        for a in gen_ints_perm(100, n):
            u = t[:a]
            u += t[a:]
            self.assertTrue(u.verify() == 1)
            self.assertTrue(equal_tree(t, u))
            u = None


# ----
# Test serialization
# ----
def revlowercmp(s1, s2):
    return cmp(string.lower(s2), string.lower(s1))


class Test_avl_pickling(unittest.TestCase):

    def setUp(self):
        self.list1 = []
        self.list2 = []

    def testfrom_iter_basic(self):
        for n in [0, 1, 10, 100, 1000, 10000]:
            a = list(range(n))
            self.assertRaises(AttributeError, avl.from_iter, a, len(a)+1)
            self.assertRaises(StopIteration, avl.from_iter, iter(a), len(a)+1)
        for n in [0, 1, 10, 100, 1000, 10000, 100000]:
            a = list(range(n))
            t = avl.from_iter(iter(a), len(a))
            self.assertTrue(verify_len(t, n))
            for j, k in gen_iter(iter(a), iter(t)):
                self.assertTrue(j == k)

    def testfrom_iter_compare(self):
        a = self.list1 + [string.lower(s) for s in self.list1]
        t = avl.new(a, unique=0, compare=revlowercmp)
        u = avl.from_iter(iter(t), len(t), compare=revlowercmp)
        self.assertTrue(u.verify() == 1)

    def testdump_basic(self):
        t = random_int_tree(0, 3000, 7000)
        for proto in [0, 2]:
            f = BytesIO()
            p = pickle.Pickler(f, proto)
            avl.dump(t, p)
            f.seek(0)
            p = pickle.Unpickler(f)
            a = avl.load(p)
            self.assertTrue(a.verify() == 1)
            self.assertTrue(equal_tree(a, t))
            f.close()

    def testdump_compare(self):
        t = avl.new(self.list1, compare=revlowercmp)
        for proto in [0, 2]:
            f = BytesIO()
            p = pickle.Pickler(f, proto)
            t.dump(p)
            f.seek(0)
            p = pickle.Unpickler(f)
            u = avl.load(p)
            self.assertTrue(u.verify() == 1)
            self.assertTrue(equal_tree(t, u))
            f.close()

    def tearDown(self):
        self.list1 = None


def suite():
    suite = unittest.TestSuite()
    for testcase in [
            Test_avl_ins,
            Test_avl_lookup,
            Test_avl_iter,
            Test_avl_dup,
            Test_avl_del,
            Test_avl_sequence,
            Test_avl_pickling]:
        suite.addTest(unittest.makeSuite(testcase, 'test'))
    return suite


if __name__ == '__main__':
    unittest.main()
