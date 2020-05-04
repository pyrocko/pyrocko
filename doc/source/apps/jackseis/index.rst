Jackseis - *manipulate seismic waveform archives*
=================================================

.. highlight:: sh

Jackseis is a simple tool to convert seismic waveform archive datasets. It can
be used to downsample, to rename meta-data, to cut out time windows, to convert
between different file formats, and to convert the sample data types.

Synopsis
--------

Jackseis is a command line tool. For example, to convert a dataset with
day-files into hour-files, you could use a command as follows::

    jackseis <input-directory> --tinc=3600 --output-dir=<output-directory>

Jackseis will recurse into the input directory and read the meta-data of all
files it can understand. It will then start converting the input files, station
by station, in chronological order. The ``--tinc=3600`` option tells it to
create files of one hour length and the ``--output-dir=...`` option to store
output to the given output directory creating files using a predefined naming
scheme.

Configuring the naming of output files
--------------------------------------

If more control on the output file naming is needed, use the
``--output=TEMPLATE`` option. With this option, quite arbitrary directory
structures can be created. For example, to pack subdirectories by year, network
and station and create files as follows:

.. code-block:: none

   2016/GE/TNTI/GE.TNTI.BHZ.2016-05-01_10-00-00.mseed
   2016/GE/TNTI/GE.TNTI.BHZ.2016-05-01_11-00-00.mseed
   2016/7G/STA1/7G.STA1.HHZ.2016-05-01_10-00-00.mseed
   2017/7G/STA1/7G.STA1.HHZ.2017-01-01_00-00-00.mseed

we would use
``--output='%(wmin_year)s/%(network)s/%(station)s/%(network)s.%(station)s.%(channel)s.%(wmin)s.mseed'``.
Jackseis will use the meta information in the waveform files to fill the
template placeholders. It will create sub-directories as neccessary. Running
Jackseis several times with the same output template will add new data to an
existing dataset. Jackseis refuses to overwrite existing files, unless the
``--force`` option is given.

Further options
---------------

Additional options exist, e.g. to downsample (``--downsample=...``), to
replace/rename meta-data (``--rename-...``), to cut out time windows
(``--tmin=...``, ``--tmax=...``), to convert between different data formats
(``--format=...``, ``--output-format=...``), and to convert the sample data
types (``--output-data-type=...``).

Run ``jackseis --help`` to find out more about any of the available options.
