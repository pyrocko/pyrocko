Dataset management - Squirrel
=============================

At the base of Pyrocko's waveform dataset handling is the
:mod:`pyrocko.squirrel` framework, centered around the
:class:`~pyrocko.squirrel.base.Squirrel` class. It indexes and caches the
meta-data of local and remote waveform datasets, however large or however many
files they are split into, and provides on-demand loading of waveform data so
that only data relevant to the current process/view has to be read from disk
(or downloaded) into the limited computer memory. It can efficiently handle
selections with millions of files. See :doc:`/topics/squirrel` for a
conceptual overview.

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

Plain format conversion, possibly combined with time-windowing, downsampling,
or restitution, does not require writing any Python code - the command line
tool `squirrel jackseis` handles this directly. See :mod:`pyrocko.io` for the
list of formats supported for output.

::

    squirrel jackseis --add mseed/ --tinc 3600 --out-format sac \
        --out-path 'sac/%(station)s_%(channel)s_%(wmin)s.sac'

If station meta-data (for example to fill in station coordinates and channel
orientation into the SAC headers) needs to be attached while converting with
:func:`pyrocko.io.save` directly, pass a dict of
:class:`~pyrocko.model.Station` objects, keyed by ``(network, station,
location)``, as its ``stations`` argument.
