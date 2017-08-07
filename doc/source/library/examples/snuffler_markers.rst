Snuffler markers
================

Snuffler's markers are a simple way to transfer interactive selections and
picks to a user script or, vice versa, to visualize processing results from a
user script in the interactive waveform browser.

Markers are handled in Pyrocko through the :py:mod:`pyrocko.marker` module.
They are represented by the :py:class:`pyrocko.marker.Marker` class and the
:py:class:`pyrocko.marker.EventMarker` and
:py:class:`pyrocko.marker.PhaseMarker` subclasses. The functions
:py:func:`pyrocko.marker.load_markers` and
:py:func:`pyrocko.marker.save_markers` can be used to read and write markers in
the file format used by Snuffler.

Read markers from file exported by Snuffler
-------------------------------------------


The following example shows how to read a Pyrocko marker file, how to
reassociate any contained phases and events, and how to access the attached
event information, which may be shared between different markers.

Download :download:`markers_example1.py </../../examples/markers_example1.py>`

.. literalinclude :: /../../examples/markers_example1.py
    :language: python
