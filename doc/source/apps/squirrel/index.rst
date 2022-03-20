
Squirrel Tool - *dataset inspection and management*
===================================================

The :program:`squirrel` command line tool is a front-end to the :ref:`Squirrel
data access infrastructure <squirrel>`. It offers functionality to

* manage separate (isolated, local) environments for different projects.
* pre-scan / index file collections.
* inspect various aspects of a data collection.
* download data from online sources (FDSN web services, earthquake catalogs).
* manage persistent selections to speed up access to very large datasets.

Basics
------

The :program:`squirrel` tool and its subcommands are self-documenting with the
``--help`` option. Run ``squirrel`` without any options to get the list of
available subcommands.  Run ``squirrel SUBCOMMAND --help`` to get details about
a specific subcommand.

Common options
--------------

Options shared between subcommands are grouped into three categories:

* **General options** include ``--loglevel`` to select the
  program's verbosity and ``--progress`` to control how progress status is
  indicated. These are provided by all of Squirrel's subcommands.

* **Data collection options** control which files and other data sources should
  be aggregated to form a dataset. The ``--add`` option to add files and
  directories. Further options are available to include/exclude files by
  regular expression patterns, to restrict to use selected content kinds only
  (waveform, station, channel, response, event), to create persistent data
  selections and more. Finally, the ``--dataset`` option is provided to
  configure the dataset conveniently in a YAML file rather than repeatedly with
  the many command line options. Using ``--dataset`` includes the possibility
  to add online data sources.

* **Data query options** are used to restrict processing/presentation to a
  subset of a data collection. They have no influence on the data collection
  itself, only on what is shown. It is possible to query by time interval
  (``--tmin``, ``--tmax``, ``--time``), channel/station code pattern
  (``--codes``), and content kinds (``--kinds``).
