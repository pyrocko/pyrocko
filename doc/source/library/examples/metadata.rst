Metadata read & write
=====================

For most day-to-day use, station and channel metadata is best obtained
through :py:meth:`~pyrocko.squirrel.base.Squirrel.get_stations` and
:py:meth:`~pyrocko.squirrel.base.Squirrel.get_channels`, from local files or
online FDSN services alike - see the :doc:`Squirrel tutorial
</library/examples/squirrel/cli_tool>`. The lower-level examples below remain
valuable for special tasks: direct StationXML file conversion and
manipulation, or building custom metadata from scratch.


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
