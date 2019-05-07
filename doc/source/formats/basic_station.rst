
.. _basic-station-files:

Basic station files
-------------------

This simple text file format can be used to hold the most basic seismic station
meta-information: station coordinates and channel orientations.

Example:

.. code-block:: none
    :caption: stations.txt

    DK.BSD.  55.11390    14.91470     88.0   0.0 Bornholm Skovbrynet, Denmark
      BHE    90     0     1
      BHN     0     0     1
      BHZ     0   -90     1
    GE.FLT1. 52.33060    11.23720    100.0   0.0
      BHE    90     0     1
      BHN     0     0     1
      BHZ     0   -90     1
    GE.RGN.  54.54770    13.32140     15.0   2.0 GRSN/GEOFON Station Ruegen
    GE.STU.  48.77190    9.19500     360.0  10.0

The file should consist of station lines optionally alternating with blocks of
channel lines. A station line has at least 5 words separated by an arbitrary
number of white-space characters:

.. code-block:: none

  <network>.<station>.<location> <lat> <lon> <elevation> <depth> <description>

* the dots in ``<network>.<station>.<location>`` are mandatory, even if some of
  the entries are empty, i.e. a station with no network and location code must
  be written as ``.STA.``
* ``<elevation>`` and ``<depth>`` must be given in [m]
* ``<description>`` is optional and may contain blanks

Channel lines have of four white-space separated words:

.. code-block:: none

  <channel> <azimuth> <dip> <gain>

* ``<azimuth>`` and ``<dip>`` define the sensor component orientation [deg],
  azimuth is measured clockwise from north, dip is measured downward from
  horizontal
* ``<gain>`` should be set to ``1``, use StationXML, SAC Pole-Zero, or RESP
  files for proper instrument response handling

Use the library function :py:func:`pyrocko.model.station.load_stations` and
:py:func:`pyrocko.model.station.dump_stations` to read and write basic station files.
