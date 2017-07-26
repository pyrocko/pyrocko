# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
# file: setup.py

from distutils.core import setup, Extension
import sys

__VERSION__ = "1.12_1"

laius = """This small C package is comprised of an independent set of
routines dedicated to manipulating AVL trees (files avl.c, avl.h), and of
an extension module for Python that builds upon it (file avlmodule.c) to
provide objects of type 'avl_tree' in Python, which can behave as sorted
containers or sequential lists.  For example one can take slices of trees
with the usual syntax.  Unlike collectionsmodule.c, avlmodule.c contains
only bindings to the underlying implementation."""

_author = "Richard McGraw"
_authoremail = "dasnar at fastmail dot fm"
_maintainer = _author
_maintaineremail = _authoremail

link_args = []
if sys.platform != 'sunos5':
    link_args.append("-Wl,-x")  # -x flag not available on Solaris

# from Distutils doc:
# patch distutils if it can't cope with "classifiers" or
# "download_url" keywords
if sys.version < '2.2.3':
    from distutils.dist import DistributionMetadata
    DistributionMetadata.classifiers = None
    DistributionMetadata.download_url = None

ext_avl = Extension(
    "avl",
    sources=["avl.c", "avlmodule.c"],
    define_macros=[('HAVE_AVL_VERIFY', None), ('AVL_FOR_PYTHON', None)],
    extra_compile_args=["-Wno-parentheses", "-Wno-uninitialized"],
    extra_link_args=link_args
    )

setup(
    name="pyavl",
    version=__VERSION__,
    description="avl-tree type for Python (C-written extension module)",
    url="http://dasnar.sdf-eu.org/miscres.html",
    download_url="http://sourceforge.net/projects/pyavl/",
    author=_author,
    author_email=_authoremail,
    maintainer=_maintainer,
    license="None, public domain",
    ext_modules=[ext_avl],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: Public Domain',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules'
        ],
    long_description=laius
    )
