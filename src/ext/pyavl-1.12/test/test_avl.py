#file "test_avl.py"

# tested with Python 2.4, september-2007 (1.12)
# tested with Python 2.5, december-2008 (1.12_1)

import sys
import unittest
import random
import string
import itertools
import StringIO
import cStringIO
import pickle
import cPickle
import avl

def verify_empty(tree):
	return tree.verify()==1 and len(tree)==0

def verify_len(tree, size):
	return tree.verify()==1 and len(tree)==size

def gen_ints(lo, hi):
	for i in xrange(lo, hi):
		yield i

# call with gcd(step,modulo)=1
def gen_ints_perm(step, modulo):
	n = random.randint(0, modulo-1)
	for i in xrange(modulo):
		val = n
		n = (val+step) % modulo
		yield val
	
def gen_pairs(lo, hi):
	for i in xrange(lo, hi):
		yield (i, 0)

def	gen_iter(a, b):
	return itertools.izip(a,b)

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

def fill_tree(tree, some_iter):
	for o in some_iter: t.insert(o)
	
# return 1 if tree1 and tree2 hold the same set of keys
def equal_tree(tree1, tree2, compare=cmp):
	if len(tree1) != len(tree2):
		return 0
	i1 = iter(tree1)
	i2 = iter(tree2)
	try:
		while compare(i1.next(), i2.next()) is 0: pass
		return 0
	except StopIteration:
		return 1
		
# sort tuples by decreasing order of first component
def pair_compare(t, u):
	if t[0] < u[0]: return +1
	if t[0] == u[0]: return 0
	return -1

#----
#Test insertions
#----
class Test_avl_ins(unittest.TestCase):
	
	def setUp(self):
		self.size = 1000
		self.list = range(self.size)
		self._times = 7
	
	# test avl.new
	def testnew_basic(self):
		for i in range(self._times):
			random.shuffle(self.list)
			t =	avl.new(self.list, unique=0)
			self.assert_(t.verify() == 1 and len(t) == self.size)
			for n in self.list: self.assert_(n in t)
		t = None
	
	# test avl.new unique=0
	def	testnew_basic_dupes(self):
		size3 = self.size * 3
		for i in range(self._times):
			random.shuffle(self.list)
			list3 = self.list * 3
			t =	avl.new(list3, unique=0)
			self.assert_(verify_len(t, size3))
			for n in self.list: self.assert_(n in t)
		t = None
	
	# test avl.new unique=1
	def	testnew_basic_unique(self):
		for i in range(self._times):
			random.shuffle(self.list)
			list3 = self.list * 3
			t =	avl.new(list3, unique=1)
			self.assert_(verify_len(t, self.size))
			for n in self.list: self.assert_(n in t)
		t = None
	
	# test avl.new with compare function
	def testnew_compare_basic(self):
		random.shuffle(self.list)
		pairs = [(i, 0) for i in self.list]
		for i in range(self._times):
			t = avl.new(pairs, compare=pair_compare)
			self.assert_(verify_len(t, self.size))
			for p in pairs: self.assert_(p in t)
	
	# test avl.new with compare function, unique=0
	def testnew_compare_dupes(self):
		random.shuffle(self.list)
		pairs = [(i, 0) for i in self.list]
		pairs3 = pairs * 3
		size3 = self.size * 3
		for i in range(self._times):
			t = avl.new(pairs3, compare=pair_compare, unique=0)
			self.assert_(verify_len(t, size3))
			for p in pairs: self.assert_(p in t)
	
	# test avl.new with compare function, unique=1
	def testnew_compare_unique(self):
		random.shuffle(self.list)
		tuples = [(i, 0) for i in self.list]
		tuples3 = tuples * 3
		for i in range(self._times):
			t = avl.new(tuples3, compare=pair_compare, unique=1)
			self.assert_(verify_len(t, self.size))
			for u in tuples: self.assert_(u in t)
	
	# comparison failure
	def testins_bad(self):
		t = avl.new([(5,6), (2,8), (7,-1), (-1,1), (2,-3)], compare=pair_compare)
		for i in range(5):
			self.assertRaises(TypeError, t.insert, 66)
			self.assertRaises(TypeError, t.has_key, 66)
			self.failIf(66 in t)
			t.insert((3,3))
			self.assert_(t.verify() == 1)
	
	# test ins and clear
	def testins_big(self):
		print 'please wait ...'
		big_size = 20000
		for i in range(self._times):
			t = random_int_tree(-big_size, big_size, big_size)
			self.assert_(verify_len(t, big_size))
			t.clear()
			self.assert_(verify_empty(t))

	def	tearDown(self):
		self.list = None

#----
#Test lookup
#----
class Test_avl_lookup(unittest.TestCase):
	
	def setUp(self):
		self.n = 5000
		self.t = range_tree(0, self.n)
		self.bad = range(-500, 0) + range(self.n, self.n+500)
		
	def testlookup_get(self):
		for i in self.bad:
			self.assertRaises(LookupError, self.t.lookup, i)
		for i in gen_ints(0, self.n):
			self.assert_(self.t.lookup(i) == i)
	
	def testlookup_contains(self):
		for i in self.bad:
			self.failIf(i in self.t or self.t.has_key(i))
		for i in gen_ints(0, self.n):
			self.assert_(i in self.t and self.t.has_key(i))
	
	def testlookup_span_one(self):
		for i in gen_ints(0, self.n):
			a,b = self.t.span(i)
			self.assert_(b-a == 1)
	
	def testlookup_span_two(self):
		t = self.t
		for i in gen_ints(0, self.n):
			j = random.randint(i, self.n-1)
			self.assert_(t.span(i,j) == (i,j+1))
			self.assert_(t.span(j,i) == (i,j+1))
			self.assert_(t.span(i,self.n) == (i,self.n))
			self.assert_(t.span(i,self.n*2) == (i,self.n))
			self.assert_(t.span(-j, i) == (0,i+1))
			
	def tearDown(self):
		self.t = None
		self.bad = None
		
#----
#Test iterations
#----
class Test_avl_iter(unittest.TestCase):
	
	def setUp(self):
		self.n = 1000
		self.t = range_tree(0, self.n)
		self.orig = range(self.n)
	
	def testiter_forloop(self):
		list = self.orig[:]
		for i in range(5):
			random.shuffle(list)
			for k in avl.new(list):
				self.assert_(k == self.orig[k])
	
	def testiter_forward(self):
		j = self.t.iter()
		for k in gen_ints(0, self.n):
			self.assert_(j.next() == k and j.index() == k and j.cur() == k)
		self.assertRaises(StopIteration, j.next)
		self.assertRaises(avl.Error, j.cur)
		self.assert_(j.index() == self.n)
		
	def testiter_backward(self):
		j = self.t.iter(1)
		for k in gen_ints(1, self.n+1):
			self.assert_(j.prev()+k == self.n and j.index()+k == self.n and j.cur()+k == self.n)
		self.assertRaises(StopIteration, j.prev)
		self.assertRaises(avl.Error, j.cur)
		self.assert_(j.index() == -1)
	
	def testiter_basic(self):
		t = avl.new()
		j = iter(t)		# before first
		k = t.iter(1)	# after last
		self.assertRaises(StopIteration, j.next)
		self.assertRaises(StopIteration, j.prev)
		self.assertRaises(StopIteration, k.next)
		self.assertRaises(StopIteration, k.prev)
		t.insert('bb')
		self.assertRaises(StopIteration, j.prev)
		self.assertRaises(StopIteration, k.next)
		self.assert_(j.next() == 'bb')
		self.assert_(k.prev() == 'bb')
		self.assertRaises(StopIteration, j.next)
		self.assertRaises(StopIteration, k.prev)
		self.assert_(j.prev() == 'bb')
		self.assert_(k.next() == 'bb')
		self.assert_(j.cur() == 'bb' and k.cur() == 'bb')
		t.insert('aa')
		self.assert_(j.prev() == 'aa')
		t.insert('cc')
		self.assert_(k.next() == 'cc')
	
	def testiter_remove(self):
		print 'please wait ...'
		for start in range(1, self.n+1):
			u = avl.new(self.t)
			self.assert_(u.verify() == 1)
			j = iter(u)
			for i in range(start): j.next()
			index = j.index()
			self.assert_(index == start-1)
			while index < len(u):
				j.remove()
				#self.assert_(u.verify() == 1)
				self.assert_(j.index() == index)
			self.assertRaises(avl.Error, j.remove)
			self.assert_(u.verify() == 1)
			
	def tearDown(self):
		self.t.clear()
		self.t = None
		self.orig = None

#----
#Test duplication
#----
class Test_avl_dup(unittest.TestCase):
	
	def setUp(self):
		self.n = 5000
		
	def testdup_basic(self):
		for i in range(10):
			t = random_int_tree(0, self.n, size=self.n)
			u = avl.new(t)
			v = t[:]
			self.assert_(u.verify() == 1)
			self.assert_(equal_tree(t, u))
			self.assert_(v.verify() == 1)
			self.assert_(equal_tree(t, v))
		t = avl.new()
		u = avl.new(t)
		self.assert_(verify_empty(u))
	
	def testdup_compare(self):
		for i in range(5):
			t = random_pair_tree(0, self.n, size=self.n, compare=pair_compare)
			u = avl.new(t)
			v = t[:]
			self.assert_(u.verify() == 1)
			self.assert_(equal_tree(t, u, pair_compare))
			self.assert_(v.verify() == 1)
			self.assert_(equal_tree(t, v, pair_compare))
		t = avl.new(compare=pair_compare)
		u = avl.new(t)
		self.assert_(verify_empty(u))		
		
#----
#Test deletions
#----
class Test_avl_del(unittest.TestCase):
	
	def setUp(self):
		pass
	
	# empty tree with unique keys
	def testdel_basic(self):
		t = avl.new()
		t.remove(1)
		t.remove(-2)
		self.assert_(verify_empty(t))
		t = range_tree(-2000, +2000)
		self.assert_(t.verify() == 1)
		n = len(t)
		# no-op
		others = range(-2100, -2000) + range(2000,2100)
		random.shuffle(others)
		for i in others: t.remove(i)
		self.assert_(verify_len(t, n))
		others = None
		# empty trees
		list = range(-2000, 2000)
		for i in range(10):
			random.shuffle(list)
			u = avl.new(t)
			for k in list: 
				u.remove(k)
				self.failIf(k in u)
			self.assert_(verify_empty(u))
	
	# empty tree with duplicates
	def testdel_lessbasic(self):
		n = 1000
		t = avl.new()
		for i in range(3):
			for k in gen_ints(0, n):
				t.insert(k)
		self.assert_(t.verify() == 1)
		# params for sequence
		modulo = n
		step = 37
		for i in range(10):
			u = avl.new(t)
			for k in gen_ints_perm(step, modulo):
				while k in u:
					u.remove(k)
				self.failIf(k in u)
			self.assert_(u.verify() == 1)
			self.assert_(len(u) == 0, 'len='+str(len(u))+' '+str(u))

	def testdel_bigunique(self):
		t = range_tree(-10000, 10000)
		for i in gen_ints_perm(10007, 20000):
			j = i - 10000
			self.assert_(t.has_key(j))
			t.remove(j)
			self.failIf(t.has_key(j))
	
	def testdel_one(self):
		print 'please wait ...'
		n = 4000
		t = random_int_tree(0, n, n)
		for i in gen_ints_perm(1993, n):
			e = t[i]
			a1, a2 = t.span(e)
			t.remove(e)
			self.assert_(t.verify() == 1)
			b1, b2 = t.span(e)
			self.assert_(a2-a1-1 == b2-b1)
			t.insert(e)
		
#----
#Test sequence support
#----
class Test_avl_sequence(unittest.TestCase):
	
	def geti(self, t, i):
		return t[i]
	
	def testseq_basic(self):
		t = avl.new()
		for step, modulo in [(491, 2000), (313, 10000)]:
			for i in gen_ints_perm(step, modulo):
				t.insert(i)
			for i in gen_ints(0, modulo):
				self.assert_(t.index(i) == i)
				self.assert_(t[i] == i)
				self.assert_(t[-i-1] == t[modulo-i-1])
			for i in gen_ints(-100, 0):
				self.assert_(t.index(i)	== -1)
			for i in gen_ints(modulo, modulo+100):
				self.assert_(t.index(i)	== -1)
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
			self.assert_(verify_len(t, n*3))
			for	i in gen_ints(0, n):
				self.assert_(t.index(i) == i*3)
				a, b = t.span(i)
				self.assert_((a == i*3) and (b-a == 3))
			t.clear()
			
	def testseq_insert(self):
		n = 5000
		t = avl.new()
		self.assertRaises(IndexError, t.insert, 'out', 1)
		self.assertRaises(IndexError, t.insert, 'out', -13)
		for i in gen_ints_perm(373, n):
			t.insert(2*i)
		self.assert_(t.verify() == 1)
		for	i in range(3):
			u = avl.new(t)
			for k in gen_ints(0, n):
				odd = 2*k+1
				u.insert(odd, odd)
				self.assert_(u[odd] == odd)
		self.assert_(u.verify() == 1)
		self.assert_(len(u) == 2*n)
		for i in range(100):
			u.append(2*n)
		for i in range(100):
			u.insert(0, 0)
		self.assert_(u.verify() == 1)
		list = None
	
	def testseq_remove(self):
		n = 5000
		t =	range_tree(0, n)
		self.assertRaises(IndexError, t.remove_at, -n-random.randint(0,100))
		for i in gen_ints_perm(29, n):
			t.remove_at(i)
			self.failIf(i in t)
			t.insert(i, i)
		self.assert_(verify_len(t, n))
		# empty the tree
		for i in gen_ints(0, n):
			k = random.randint(0, len(t)-1)
			e = t[k]
			t.remove_at(k)
			self.failIf(e in t or t.index(e) != -1)
		self.assert_(verify_empty(t))
		self.assertRaises(IndexError, t.remove_at, 0)
		
	def testseq_spaninsert(self):
		print 'please wait ...'
		n =	5000
		t = avl.new(compare=lambda x,y:cmp(y,x))
		for i in range(2):
			for	i in gen_ints(0, 3*n):
				e = random.randint(-n,n)
				a1, a2 = t.span(e)
				t.insert(e,	a2)
				self.assert_(t.span(e) == (a1,a2+1))
			self.assert_(verify_len(t, 3*n))
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
			self.assert_(b-a == repeats)
			self.assert_(a==0 or t[a-1] < k)
			self.assert_(b==len(t) or t[b] > k)
			for i in xrange(repeats): t.remove_at(a)
			self.assert_(t.span(k) == (a,a))
		self.assert_(verify_empty(t))
	
	def testseq_slicedup(self):
		n = 3000
		for i in xrange(3):
			t = random_int_tree(0, n, 4*n)
			self.assert_(equal_tree(t[:], t))
	
	def testseq_sliceempty(self):
		t = random_int_tree(0, 500, size=1000)
		lim = len(t)
		for i in xrange(100):
			a = random.randint(0, lim)
			b = random.randint(0, a)
			self.assert_(verify_empty(t[a:b]))
			a = random.randint(1, lim)
			b = random.randint(1, a)
			self.assert_(verify_empty(t[-b:-a]))
		
	def testseq_slice(self):
		print 'please wait ...'
		n = 1000
		t = range_tree(0, n)
		for a in xrange(n):
			u = t[:a]
			self.assert_(verify_len(u, a))
			self.assert_(equal_tree(u, range_tree(0,a)))
			u = t[a:]
			self.assert_(equal_tree(u, range_tree(a,n)))
		t = None
		
	#test a+b
	def testseq_sliceconcat(self):
		print 'please wait ...'
		n = 2000
		e = avl.new()
		t =	range_tree(0, n)
		self.assert_(verify_empty(e+e))
		self.assert_(equal_tree(t+e, t))
		self.assert_(equal_tree(e+t, t))
		for a in gen_ints_perm(50, n):
			u = t[:a] + t[a:]
			self.assert_(u.verify() == 1)
			self.assert_(equal_tree(t, u))
			u = None
			
	def testseq_concatinplace(self):
		self.assertRaises(TypeError, avl.new().concat, [])
		n = 2000
		e = avl.new()
		t = range_tree(0, n)
		u = t[:]
		u.concat(e)
		self.assert_(equal_tree(t, u))
		e.concat(u)
		self.assert_(equal_tree(e, u))
		u += avl.new()
		self.assert_(equal_tree(t, u))
		e.clear()
		e += t
		self.assert_(equal_tree(e, t))
		e.clear()
		for a in gen_ints_perm(100, n):
			u = t[:a]
			u += t[a:]
			self.assert_(u.verify() == 1)
			self.assert_(equal_tree(t, u))
			u = None
		
#----
#Test serialization
#----
def revlowercmp(s1, s2):
	return cmp(string.lower(s2), string.lower(s1))

class Test_avl_pickling(unittest.TestCase):
	
	def setUp(self):
		self.list1 = []
		self.list2 = []
		
	def testfrom_iter_basic(self):
		for n in [0, 1, 10, 100, 1000, 10000]:
			a = xrange(n)
			self.assertRaises(AttributeError, avl.from_iter, a, len(a)+1)
			self.assertRaises(StopIteration, avl.from_iter, iter(a), len(a)+1)
		for n in [0, 1, 10, 100, 1000, 10000, 100000]:
			a = xrange(n)
			t = avl.from_iter(iter(a), len(a))
			self.assert_(verify_len(t, n))
			for j, k in gen_iter(iter(a), iter(t)):
				self.assert_(j == k)
	
	def testfrom_iter_compare(self):
		a = self.list1 + map(lambda s:string.lower(s), self.list1)
		t = avl.new(a, unique=0, compare=revlowercmp)
		u = avl.from_iter(iter(t), len(t), compare=revlowercmp)
		self.assert_(u.verify() == 1)
		
	def testdump_basic(self):
		t = avl.new(self.list1)
		for proto in [0,2]:
			f = StringIO.StringIO()
			p = pickle.Pickler(f, proto)
			t.dump(p)
			f.seek(0)
			p = pickle.Unpickler(f)
			a = avl.load(p)
			self.assert_(a.verify() == 1)
			self.assert_(equal_tree(a, t))
			f.close()
		t = random_int_tree(0, 3000, 7000)
		for m in [pickle, cPickle]:
			for proto in [0, 2]:
				f = cStringIO.StringIO()
				p = m.Pickler(f, proto)
				avl.dump(t, p)
				f.seek(0)
				p = pickle.Unpickler(f)
				a = avl.load(p)
				self.assert_(a.verify() == 1)
				self.assert_(equal_tree(a, t))
				f.close()
		
		
	def	testdump_compare(self):
		t = avl.new(self.list1, compare=revlowercmp)
		for proto in [0, 2]:
			f = StringIO.StringIO()
			p = pickle.Pickler(f, proto)
			t.dump(p)
			f.seek(0)
			p = pickle.Unpickler(f)
			u = avl.load(p)
			self.assert_(u.verify() == 1)
			self.assert_(equal_tree(t, u))
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

def main():
	print sys.version
	random.seed()
	unittest.TextTestRunner(verbosity=2).run(suite())
	
if __name__ == '__main__':
	main()
	