Downloading seismic data (FDSN)
================================

Waveforms and meta data can be retrieved from online `FDSN services <http://www.fdsn.org>`_. The recommended way to do so is through the :mod:`pyrocko.squirrel` framework, using :py:meth:`~pyrocko.squirrel.base.Squirrel.add_fdsn` to declare an FDSN web service as a data source; Squirrel uses the lower-level :py:mod:`pyrocko.client.fdsn` module under the hood.


Downloading and restituting waveform data
------------------------------------------

Downloading waveform data and instrument response information, and using the
latter to deconvolve the transfer function from the waveform traces, is
covered step by step in the :doc:`Squirrel tutorial
</library/examples/squirrel/cli_tool>` (see the ``squirrel_rms1.py`` and
``squirrel_rms2.py`` examples there).

For a one-off download where Squirrel's indexing and caching machinery would
be overkill, the lower-level :py:mod:`pyrocko.client.fdsn` module can be used
directly instead:

::

    from pyrocko.client import fdsn
    from pyrocko import io, util

    tmin = util.stt('2014-01-01 16:10:00.000')
    tmax = util.stt('2014-01-01 16:39:59.000')

    selection = [
        ('GE', 'EIL', '*', '*Z', tmin, tmax),   # all vertical components
    ]

    request_waveform = fdsn.dataselect(site='geofon', selection=selection)

    with open('traces.mseed', 'wb') as file:
        file.write(request_waveform.read())

    traces = io.load('traces.mseed')


StationXML data manipulation
----------------------------

To manipulate `StationXML <http://www.fdsn.org/xml/station/>`_ data through
Pyrocko use the :py:mod:`pyrocko.io.stationxml` module.  This example will
change the azimuth and dip values for channels whose codes are X, Y and Z, and
set all channel instrument's input units to meters.


 .. literalinclude :: /../../examples/fdsn_stationxml_modify.py
    :language: python

Download :download:`fdsn_stationxml_modify.py </../../examples/fdsn_stationxml_modify.py>`
