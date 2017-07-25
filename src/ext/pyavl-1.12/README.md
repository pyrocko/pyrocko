/* AVL tree type for Python */

WHAT IS IT ?
-----------------------
This small C package is made of an independent set of routines
dedicated to manipulating AVL trees (files avl.c, avl.h), and of an
extension module for Python that builds upon it (file avlmodule.c).  Unlike
collectionsmodule.c, the latter file contains only Python bindings: it adds
nothing to the underlying implementation.

TERMS OF USE: This package, pyavl, is donated to the public domain.

Author: Richard McGraw
Email:  dasnar@fastmail.fm

ACKNOWLEDGMENTS
-----------------------
Sam Rushing's own Python extension module based on an iterative C
implementation, is to be found at
	<http://www.python.org/ftp/python/contrib-09-Dec-1999/DataStructures/avl.tar.gz>

Ben Pfaff is responsible for the outstanding C library `libavl' to be found
in GNU ftp archives.  If you are not interested in Python bindings or in
indexed access, you are likely better off with libavl.  Check Ben Pfaff's
pages at:
	<http://adtinfo.org/>

COMPILING
-----------------------
To build the extension module, run 'python setup.py install'
See README.MPW on how to make it in Apple MPW.

To test the raw C part, run 'make -f test.make test_avl'
(does not test avl_slice nor avl_xload)

For a better test, run 'test/test_avl.py' :)

NOTES AND LIMITATIONS
-----------------------
An avl_tree is a set, not a map. 
PyMem_Malloc() is called every time an item is inserted into a tree.  This
is inefficient.
The module defines iterators, a new construct in Python 2.2: hence
Python 2.2 or later is required in order to compile with support for the
iterator protocol.
WARNING (see the pdf file): calling tree.remove(o) while some
iter points to o is unsafe (possible crash).

MANUAL
-----------------------
See pyavl.tex, pyavl-a4.pdf

CHANGES
-----------------------
    2017/07
Added Python 3 support to pyavl.
Thanks to Sebastian Heimann for realisation
	2008/12
version number 1.12_1
tested on Mac OS 10.5.6 and macports Python 2.5 (with python setup.py)
corrected avl_tree_repr (in avlmodule.c) which was very slow (see http://sourceforge.net/tracker/index.php?func=detail&aid=1941028&group_id=125222&atid=701866) 
	2007/09
code correction by David Turner <novalis@openplans.org>: see avl_iterator_kill in avl.c
	2006/05
fixed problem with compilation of avlmodule.c on windows (initializers)
thanks to Dennis Follensbee for reporting the problem
	2005/11/23
corrected mistake for module functions that want to return (-1)
	2005/09/25
compiled on Mac OS 10.3.9 (with gcc 3.3), tested with Python 2.4 (Fink package)
	2004/12/10
corrected error in concat_inplace and concat_inplace_seq code which caused
a SystemError to be raised when right-hand side tree was empty
	2004/12
compiled in MPW (MPW 3.6d8, Mac OS 9.2, MrC 5.0d3c1)
added unit tests
	2004/11
version number 1.1
added a factory function (avl.from_iter) and convenience functions for
serialization (pickling)
	2004/10/29
compiled on Mac OS 10.3 (with gcc 3.3), tested with Python 2.3
	2004/08/21
prepared a better MPW Makefile (see README.MPW)
	2004/01/26
fixed programming error in avl.c: avl_dup and avl_slice like dummy copied
t->param !


Comments are welcome.
