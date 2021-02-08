``squirrel``
============

Prompt seismological data access with a fluffy tail.

Features

- Efficient[1] lookup of data relevant for a selected time window.
- Metadata caching and indexing.
- Modified files are re-indexed as needed.
- SQL database (sqlite) is used behind the scenes.
- Can handle selections with millions of files.
- Data can be added and removed at run-time, efficiently[1].
- Just-in-time download of missing data.
- Disk-cache of meta-data query results with expiration time.
- Efficient event catalog synchronization.
- Always-up-to-date data coverage indices.
- Always-up-to-date indices of available station/channel codes.

[1] O log N performance, where N is the number of data entities (nuts).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   base <base>
   model <model>
   pile <pile>
   environment <environment>
   error <error>
   io <io/index>
   client <client/index>
   dataset <dataset>
   cache <cache>
