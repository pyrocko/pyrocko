Dataset management - The ``pile``
==================================

At the base of Pyrocko's waveform dataset handling is the
:class:`~pyrocko.pile.Pile` class. It organizes and caches the meta-data of
large waveform datasets split into many files and provides on-demand loading of
waveform data so that only data relevant to the current process/view has to be
read from disk into the limited computer memory. It can efficiently handle up
to a few 100000 files for interactive processes.

Reorganizing a dataset into hour-files
--------------------------------------
In each iteration we get all data for the current time window as a
:py:class:`~pyrocko.squirrel.base.Batch`, yielded by
:py:meth:`pyrocko.squirrel.base.Squirrel.chopper_waveforms`. The batch 'knows'
the time window to which it belongs, stored in the attributes ``batch.tmin``
and ``batch.tmax`` (note: a trace's own onset, ``trace.tmin``, does not have
to be identical to ``batch.tmin``). The directory parts in the output path
will be created as necessary.
When applying this procedure to a dataset consisting of arbitrarily separated
files, it will automatically connect adjacent traces as needed!


.. literalinclude :: /../../examples/squirrel_hour_files.py
    :language: python

Download :download:`squirrel_hour_files.py </../../examples/squirrel_hour_files.py>`


Downsampling a whole dataset
----------------------------

Example for downsampling all trace files in the input folder to a common sampling rate with :py:meth:`pyrocko.trace.Trace.downsample_to`. Padding (``tpad``) is added around each window and trimmed off again after downsampling, to absorb filter edge effects at window boundaries.

.. literalinclude :: /../../examples/squirrel_downsample.py
    :language: python

Download :download:`squirrel_downsample.py </../../examples/squirrel_downsample.py>`


Converting a dataset from Mini-SEED to SAC format
-------------------------------------------------

Conversion of a mseed file to SAC. See :mod:`pyrocko.io` for supported formats.

.. literalinclude :: /../../examples/convert_mseed_sac.py
    :language: python

Download :download:`convert_mseed_sac.py </../../examples/convert_mseed_sac.py>`
