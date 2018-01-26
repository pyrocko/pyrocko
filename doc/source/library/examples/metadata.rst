Metadata read & write
=====================

QuakeML import
--------------

This example shows how to read quakeml-event catalogs using :func:`~pyrocko.model.quakeml.QuakeML_load_xml()`.
The function :meth:`~pyrocko.model.quakeml.QuakeML.get_pyrocko_events()` is used to obtain events in pyrocko format.
If a moment tensor is provided as [``Mrr, Mtt, Mpp, Mrt, Mrp, Mtp``], this is converted to [``mnn, mee, mdd, mne, mnd, med``]. The strike, dip and rake values appearing in the pyrocko event are calculated from the moment tensor.

.. literalinclude :: /../../examples/readnwrite_quakml.py
    :language: python
