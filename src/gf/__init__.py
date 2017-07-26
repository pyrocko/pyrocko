# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
'''
A Green's function storage and synthetic seismogram synthesis framework.

The :mod:`pyrocko.gf` subpackage splits functionality into several submodules:

* The :mod:`pyrocko.gf.store` module deals with the storage, retrieval and
  summation of Green's functions.
* The :mod:`pyrocko.gf.meta` module provides data structures for the meta
  information associated with the Green's function stores and implements
  various the Green's function lookup indexing schemes.
* The :mod:`pyrocko.gf.builder` module defines a common base for Green's
  function store builders.
* The :mod:`pyrocko.gf.seismosizer` module provides high level synthetic
  seismogram synthesis.

All classes defined in the :mod:`pyrocko.gf.*` submodules are imported into the
:mod:`pyrocko.gf` namespace, so user scripts may simply use ``from pyrocko
import gf`` or ``from pyrocko.gf import *`` for convenience.

'''
from __future__ import absolute_import

from .meta import *  # noqa
from .store import *  # noqa
from .builder import *  # noqa
from .seismosizer import *  # noqa
from .targets import *  # noqa
