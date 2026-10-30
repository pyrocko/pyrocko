
.. image:: /static/squirrel.svg
   :align: left

Squirrel command line tool manual
=================================

The :program:`squirrel` command line tool is a front-end to the :doc:`Squirrel
data access infrastructure </topics/squirrel>`.

.. raw:: html

   <div style="clear:both"></div>

It offers functionality to

* inspect various aspects of a data collection.
* pre-scan / index file collections.
* download data from online sources (FDSN web services, earthquake catalogs).
* convert large data collections (file format and directory layout)
* manage separate (isolated, local) environments for different projects.
* manage persistent selections to speed up access to very large datasets.

Please read the :doc:`Tutorial <tutorial>` to get started with the ``squirrel``
command line tool.

Command reference
-----------------

The :app:`squirrel` tool and its subcommands are self-documenting with the
``--help`` option. Run ``squirrel`` without any options to get the list of
available subcommands.  Run ``squirrel SUBCOMMAND --help`` to get details about
a specific subcommand, e.g. ``squirrel scan --help``.

.. toctree::
   :maxdepth: 3

   squirrel <reference/squirrel>

.. _squirrel_common_options:

Common options
--------------

Options shared between subcommands are grouped into three categories:

* **General options** include ``--loglevel`` to select the
  program's verbosity and ``--progress`` to control how progress status is
  indicated. These are provided by all of Squirrel's subcommands.

* **Data collection options** control which files and other data sources should
  be aggregated to form a dataset. Use the ``--add`` option to add files and
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

.. note::

    Squirrel's codes have a fifth element, the *extra* code, in addition to
    network, station, location and channel (``NET.STA.LOC.CHA.EXTRA``). It
    is empty for ordinary recordings but is used e.g. to label processed or
    derived data. A ``--codes`` pattern with four elements only matches codes
    with an empty extra code: ``GR.GRA1..BHZ`` does not match
    ``GR.GRA1..BHZ.MS``. To match any extra code, append ``.*``, e.g.
    ``--codes 'GR.GRA1..BHZ.*'``. Squirrel tools print a hint when a pattern
    matches nothing for this reason.

.. note::

    Mini-SEED cannot store the extra code, and it limits the length of the
    other codes to 2 (network), 5 (station), 2 (location) and 3 (channel)
    characters. When writing Mini-SEED, the extra code is dropped. Codes
    that are too long cause an error.

Examples
--------

.. contents :: Content
  :local:
  :depth: 4

Check data integrity
....................

I am not sure if my Mini-SEED waveforms and StationXML metadata are properly
organized and if the station epochs are correctly specified. How do I check the
integrity of my dataset?

::

  squirrel check --add data/raw meta/stations.xml

Data inventory
..............

My professor gave me a huge collection of waveforms and metadata. How can
I get an overview on the available data?

Get overview on time spans of waveforms, channels, stations, responses::

  squirrel coverage --add data/

List available codes::

  squirrel codes --add data/

List some details about all data entities of kind ``channel``::

  squirrel nuts --add data/ --kind channel --style summary

List all files containing waveforms from station XYZ::

  squirrel files --add data/ --codes '*.XYZ.*.*' --kind waveform

List all data entities matching a given time span::

  squirrel nuts --add data/ --tmin '2020-02-22 10:00' --tmax '2020-02-23 05:00'

Renaming waveform channel codes
...............................

I have waveforms with channels named ``p0``, ``p1`` and ``p2``. I would like to
rename them to ``HHZ``, ``HHN``, and ``HHE``, respectively::

  squirrel jackseis --add data/old --out-sds-path data/new --rename-channel 'p0:HHZ,p1:HHN,p2:HHE'

This will create a new directory hierarchy in SDS layout under ``data/new``
with the modified waveforms.

For further possibilities, look for ``--rename-channel`` in the output of
``squirrel jackseis --help``.

Set network code on a set of waveforms
......................................

I have waveforms where the network code is partially missing and partially
incorrect. I would like to set it for all waveforms to XX::

  squirrel jackseis --add data/old --out-sds-path data/new --rename-network XX

Convert raw waveforms to instrument-corrected ground velocity
.............................................................

I would like to convert raw waveforms to ground velocity in m/s. I have
continuous waveforms and I don't want to have any gaps in the output::

  squirrel jackseis --add data/raw meta/stations.xml --out-sds-path data/velocity \
     --quantity velocity --band 0.1,10

This will restitute the seismograms to ground velocity in the frequency band
0.1 to 10 Hz. It uses :py:meth:`pyrocko.trace.Trace.transfer` under the hood.
The frequency taper involved is flat between 0.1 and 10 Hz and decays to zero
at 0.1/2 and 10*2 Hz.

Reducing memory consumption when converting data with ``squirrel jackseis``
...........................................................................

Data is processed in time blocks matching the time increment given with
``--tinc`` (plus padding). All available data in the current block is loaded
into memory. This may lead to excessive memory usage when lots of channels are
available in the dataset.

Add ``--traversal sensor`` to process the dataset sensor by sensor. It will run
a loop over time for each sensor in the dataset and thus use less memory.

Convert DiGOS/Omnirecs DATA-CUBE recordings to MiniSEED
.......................................................

I want to convert some continuous DATA-CUBE recordings into Mini-SEED format.
The output should be in SDS structure::

  squirrel jackseis --add data/data-cube --out-sds-path data/mseed --rename-network XX

This will resample the data with sinc interpolation using the GPS information
in the raw data. Add ``--rename-station ...``, ``--rename-channel ...`` as
needed.

**Note:** Always give the complete recording directory as input as the GPS
information from neighboring files may be used for an optimal time correction.

Downsampling
............

I want to create a downsampled variant of my dataset::

  squirrel jackseis --add data/raw --out-sds-path data/raw-10hz --out-data-type int32 --downsample 10

The output will be sampled at 10 Hz and the datatype in the Mini-SEEDs will be
32-bit integers. Without the ``--out-data-type`` option, the output would be in
double-precision floating point format.

Spectrograms of long recordings
...............................

*Experimental.* I would like to look at spectrograms of several months of
continuous recordings. With the waveforms in ``data/raw`` and the station and
response metadata in ``meta/stations.xml``, compute the spectrograms::

  squirrel spectrogram --add data/raw meta/stations.xml --out-storage-path data/spectrograms/main

The instrument response is removed, so the spectrograms show ground velocity
(see ``--quantity``). To browse long time spans quickly, pre-compute
downsampled overviews::

  squirrel cascade --add data/spectrograms/main --out-storage-path data/spectrograms/cascade

Then view spectrograms and overviews in the web browser (see below)::

  squirrel service --add data/spectrograms

Spectrograms and overviews are stored as *carpets* (see
:py:mod:`pyrocko.carpet`), in files organized by codes and time.

Explore a dataset in the web browser
....................................

*Experimental.* I would like to interactively browse the waveforms, coverage,
station metadata and events of my dataset::

  squirrel service --add data/raw meta/stations.xml

This starts a local web server and prints the address to open in the browser.
Waveforms, carpets and their coverage are shown on a timeline, stations on a
map. Processed data, e.g. instrument-corrected waveforms, can be added with
``--mantra``, see :doc:`/topics/squirrel_mantras`. Run ``squirrel service
--help-port-forwarding`` for how to access a service running on a remote
machine.

Collect events in a local event database
........................................

*Experimental.* I want to keep a local copy of the events of a catalog, which I
can update from time to time, e.g. for all events of magnitude 6 and above in
the Global-CMT catalog::

  squirrel catalog --dataset :events-gcmt-m6 --tmin 2020 --tmax 2025 --out events.elk

Running the command again later adds new events and updates revised ones. The
database file can then be used like any other event source::

  squirrel catalog --add events.elk --tmin 2024 --tmax 2025
