ObsPy Compatibility
===================

Pyrocko is compatible with `ObsPy <https://obspy.org>`_'s data model. The compatibility package :mod:`pyrocko.obspy_compat` can be activated easily to install additional methods on Pyrocko and ObsPy objects. This enables easy conversion between waveform data as well as station and event information between the two projects.

.. code-block:: python

    import obspy
    from pyrocko import obspy_compat
    obspy_compat.plant()


Using the Snuffler on ``obspy.Trace``
-------------------------------------

This is how you use :meth:`~pyrocko.trace.Trace.snuffle` on an :class:`obspy.Trace  <obspy.core.trace.Trace>`.

 .. literalinclude :: /../../examples/obspy_compat_snuffle.py
    :language: python

Download :download:`obspy_compat_snuffle.py </../../examples/obspy_compat_snuffle.py>`

List of installed methods
-------------------------

When :func:`pyrocko.obspy_compat.plant` is executed several new methods are attached to Pyrocko and ObsPy classes.

+--------------------------------------+---------------------------------+
| class                                | methods                         |
+======================================+=================================+
| :py:class:`obspy.Trace`              | :py:func:`to_pyrocko_trace`     |
|                                      +---------------------------------+
|                                      | :py:func:`snuffle`              |
|                                      +---------------------------------+
|                                      | :py:func:`fiddle`               |
+--------------------------------------+---------------------------------+
| :py:class:`obspy.Stream`             | :py:func:`to_pyrocko_traces`    |
|                                      +---------------------------------+
|                                      | :py:func:`snuffle`              |
|                                      +---------------------------------+
|                                      | :py:func:`fiddle`               |
+--------------------------------------+---------------------------------+
| :py:class:`obspy.Catalog`            | :py:func:`to_pyrocko_events`    |
+--------------------------------------+---------------------------------+
| :py:class:`obspy.Inventory`          | :py:func:`to_pyrocko_stations`  |
+--------------------------------------+---------------------------------+

Methods added to Pyrocko classes are:

+--------------------------------------+---------------------------------+
| class                                | methods                         |
+======================================+=================================+
| :py:class:`pyrocko.trace.Trace`      | :py:func:`to_obspy_trace`       |
+--------------------------------------+---------------------------------+
| :py:class:`pyrocko.pile.Pile`        | :py:func:`to_obspy_traces`      |
+--------------------------------------+---------------------------------+


