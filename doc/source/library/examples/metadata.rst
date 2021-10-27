Metadata read & write
=====================


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
