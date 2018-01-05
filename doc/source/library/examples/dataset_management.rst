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
In each iteration we get all data for the current time window as a list of traces. The traces emitted by :py:meth:`pyrocko.pile.Pile.chopper()` 'know' the time window to which they belong; it is stored in the attributes ``trace.wmin`` and ``trace.wmax``.
note: ``trace.tmin`` (its onset) does not have to be identical to ``trace.wmin``. The directory parts in the output path will be created as neccessary.
When applying this procedure to a dataset consisting of arbitrarily separated files, it will automatically connect adjacent traces as needed!
The :mod:`time` and :mod:`calendar` modules will be used.


.. literalinclude :: /../../examples/make_hour_files.py
    :language: python

Download :download:`catalog_search_globalcmt.py </../../examples/make_hour_files.py>`


Downsampling a whole dataset
----------------------------

Example for downsampling all trace files in the input folder to a common sampling rate with :py:meth:`pyrocko.trace.Trace.downsample_to`.

.. literalinclude :: /../../examples/pile_downsample.py
    :language: python

Download :download:`pile_downsample.py </../../examples/pile_downsample.py>`


Converting a dataset from Mini-SEED to SAC format
-------------------------------------------------

Conversion of a mseed file to SAC. See :mod:`pyrocko.io` for supported formats.

.. literalinclude :: /../../examples/convert_mseed_sac.py
    :language: python

Download :download:`convert_mseed_sac.py </../../examples/convert_mseed_sac.py>`
