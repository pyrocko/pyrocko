
.. image:: /static/squirrel.svg
   :align: left

.. module :: pyrocko.squirrel

``squirrel``
============

Prompt seismological data access with a fluffy tail.

.. raw:: html

   <div style="clear:both"></div>

This is the reference manual for the Squirrel framework. It describes the API
of the :py:mod:`pyrocko.squirrel` subpackage. For a conceptual overview see
:doc:`/topics/squirrel`. A tutorial describing the API usage is provided at
:doc:`Examples: Squirrel powered data access </library/examples/squirrel/index>`.

**Usage**

Public symbols implemented in the various submodules are aggregated into the
``pyrocko.squirrel`` namespace for use in user programs::

    from pyrocko.squirrel import Squirrel

    sq = Squirrel()

**Implementation overview**

The central class and interface of the framework is
:py:class:`~pyrocko.squirrel.base.Squirrel`, part of it is implemented in its
base class :py:class:`~pyrocko.squirrel.selection.Selection`. Core
functionality directly talking to the database is implemented in
:py:mod:`~pyrocko.squirrel.base`, :py:mod:`~pyrocko.squirrel.selection` and
:py:mod:`~pyrocko.squirrel.database`. The data model of the framework is
implemented in :py:mod:`~pyrocko.squirrel.model`. User project environment
detection is implemented in :py:mod:`~pyrocko.squirrel.environment`. Portable
dataset description in :py:mod:`~pyrocko.squirrel.dataset`. A unified IO
interface bridging to various Pyrocko modules outside of the Squirrel framework
is available in :py:mod:`~pyrocko.squirrel.io`. Memory cache management is
implemented in :py:mod:`~pyrocko.squirrel.cache`. Compatibility with Pyrocko's
older waveform archive access module is implemented in
:py:mod:`~pyrocko.squirrel.pile`.


**Submodules**

.. toctree::
   :maxdepth: 1

   base <base>
   selection <selection>
   database <database>
   model <model>
   environment <environment>
   dataset <dataset>
   client <client/index>
   io <io/index>
   cache <cache>
   error <error>
   pile <pile>
   tool <tool>
