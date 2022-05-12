
.. image:: /static/squirrel.svg
   :align: left

.. _squirrel:

``squirrel``
============

Prompt seismological data access with a fluffy tail.

The :program:`squirrel` command line tool is a front-end to the :doc:`Squirrel
data access infrastructure </library/reference/squirrel/index>`.

The Squirrel framework provides a unified interface to query and access seismic
waveforms, station meta-data and event information from local file collections
and remote data sources. For prompt responses, a database setup is used under
the hood. To speed up assemblage of ad-hoc data selections, files are indexed
on first use and the extracted meta-data is remembered for subsequent accesses.
Bulk data is lazily loaded from disk and remote sources, just when requested.
Once loaded, data is cached in memory to expedite typical access patterns.
Files and data sources can be dynamically added and removed at runtime.

**Features**

- Efficient (O log N) lookup of data relevant to a time window of interest.
- Metadata caching and indexing.
- Modified files are re-indexed as needed.
- SQL database (sqlite) is used behind the scenes.
- Can handle selections with millions of files.
- Data can be added and removed at run-time, efficiently (O log N).
- Just-in-time download of missing data.
- Disk-cache of meta-data query results with expiration time.
- Efficient event catalog synchronization.
- Always-up-to-date data coverage indices.
- Always-up-to-date indices of available station/channel codes.

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
   :maxdepth: 2

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
