Metadata read & write
=====================

QuakeML import
--------------

This example shows how to read quakeml-event catalogs using :func:`~pyrocko.model.quakeml.QuakeML_load_xml()`.
The function :meth:`~pyrocko.model.quakeml.QuakeML.get_pyrocko_events()` is used to obtain events in pyrocko format.
If a moment tensor is provided as [``Mrr, Mtt, Mpp, Mrt, Mrp, Mtp``], this is converted to [``mnn, mee, mdd, mne, mnd, med``]. The strike, dip and rake values appearing in the pyrocko event are calculated from the moment tensor.

.. literalinclude :: /../../examples/readnwrite_quakml.py
    :language: python

StationXML import
-----------------
This example shows how to import StationXML files and extract pyrocko.Station objects

.. literalinclude :: /../../examples/station_from_XML.py
    :language: python


Pyrocko stations to StationXML
------------------------------
This example shows how to import pyrocko stations and save FDSN StationXML files.

.. literalinclude :: /../../examples/stations_pyr2xml.py
    :language: python


Create a StationXML file with flat displacement responses
---------------------------------------------------------

In this example, we read a Pyrocko basic station file, create an FDSN
StationXML structure from it and add flat reponses to all channels. The created
StationXML file could e.g. be used in combination with restituted data, to
properly indicate that we are dealing with displacement seismograms given in
[m].

.. literalinclude :: /../../examples/make_flat_stationxml.py
