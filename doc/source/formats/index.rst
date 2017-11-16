
File formats
============

Pyrocko can input and output data into various standard formats. The preferred
formats are `Mini-SEED` for seismic waveforms and `StationXML` for station
metadata. Additionaly, Pyrocko uses its own formats for certain purposes. These
are described in this section.

YAML based file formats in Pyrocko
----------------------------------

The default IO format for many of Pyrocko's internal structures and its
configuration files is the `YAML <http://yaml.org/>`_ format. A generic
mechanism is provided to allow users to define arbitrary new types with YAML IO
support. These can nest or extend Pyrocko's predefined types. The functionality
for this is provided via the :py:mod:`pyrocko.guts` module, usage examples can
be found in section :doc:`/library/examples/guts`.

For example, here is how a :py:class:`pyrocko.model.Station` object is
represented in YAML format:

.. code-block:: yaml
    :caption: station.yaml

    --- !pf.Station
    network: DK
    station: BSD
    location: ''
    lat: 55.1139
    lon: 14.9147
    elevation: 88.0
    depth: 0.0
    name: Bornholm Skovbrynet, Denmark
    channels:
    - !pf.Channel
      name: BHE
      azimuth: 90.0
      dip: 0.0
      gain: 1.0
    - !pf.Channel
      name: BHN
      azimuth: 0.0
      dip: 0.0
      gain: 1.0
    - !pf.Channel
      name: BHZ
      azimuth: 0.0
      dip: -90.0
      gain: 1.0

Though YAML or standard file formats can be used to hold station or event
information, for the sake of simplicity, two basic text file formats are
supported for stations and events. One other simple file format has been
defined to store markers from the :doc:`/apps/snuffler/index` application.
These file formats are briefly described below.

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

Use the library function :py:func:`pyrocko.model.load_stations` and
:py:func:`pyrocko.model.dump_stations` to read and write basic station files.

.. _basic-event-files:

Basic event files
-----------------

This simple text file format can be used to hold most basic earthquake catalog
information.

Example:

.. code-block:: none
    :caption: events.txt

    name = ev_1 (cluster 0)
    time = 2014-11-16 22:27:00.105
    latitude = 64.622
    longitude = -17.4295
    magnitude = 4.27346
    catalog = bardarbunga_reloc
    --------------------------------------------
    name = ev_2 (cluster 0)
    time = 2014-11-18 03:18:41.398
    latitude = 64.6203
    longitude = -17.4075
    depth = 5000
    magnitude = 4.34692
    moment = 3.7186e+15
    catalog = bardarbunga_reloc
    --------------------------------------------
    name = ev_3 (cluster 0)
    time = 2014-11-23 09:22:48.570
    latitude = 64.6091
    longitude = -17.3617
    magnitude = 4.9103
    moment = 2.60286e+16
    depth = 3000
    mnn = 2.52903e+16
    mee = 1.68639e+15
    mdd = -1.03187e+16
    mne = 9.8335e+15
    mnd = -7.63905e+15
    med = 1.9335e+16
    strike1 = 77.1265
    dip1 = 57.9522
    rake1 = -138.246
    strike2 = 321.781
    dip2 = 55.6358
    rake2 = -40.0024
    catalog = bardarbunga_mti
    --------------------------------------------

* depth must be given in [m]
* moment tensor entries must be given in [Nm], in north-east-down coordinate
  system

Use the library functions :py:func:`pyrocko.model.load_events` and
:py:func:`pyrocko.model.dump_events` to read and write basic event files.

.. note::

    The basic event file format is a relic from pre-YAML Pyrocko. Consider
    using YAML format to read/write event objects in newer applications.

.. _marker-files:

Marker files
------------

The marker file format is used to store markers and pick information from
:doc:`/apps/snuffler/index` in a simple way.

Example:

.. code-block:: none
    :caption: example.markers

    # Snuffler Markers File Version 0.2
    event: 2015-04-16 06:38:08.8350  0 4342fb5oj726   51.4177088165 12.1322880252  29344.72658 3.22029 None  gfz2015hkiy None
    event: 2017-04-29 00:56:23.3900  0 sbqqrmbj03ce   51.3385103357 12.2131631055  27253.08273 2.88913 None  gfz2017ihrf None
    phase: 2015-04-16 06:38:16.2762  0 SX.NEUB..BHZ    4342fb5oj726   2015-04-16   06:38:08.8350 P        None False
    phase: 2015-04-16 06:38:21.3077  0 SX.NEUB..BHN    4342fb5oj726   2015-04-16   06:38:08.8350 S        None False
    phase: 2015-04-16 06:38:17.6081  0 SX.WIMM..BHZ    4342fb5oj726   2015-04-16   06:38:08.8350 P        None False
    phase: 2015-04-16 06:38:27.2764 2015-04-16 06:38:28.2630 0.986566066742  0 TH.ABG1..BHZ    4342fb5oj726   2015-04-16   06:38:08.8350 S        None False
    2015-04-16 06:38:13.9964  0 TH.CHRS..BHE
    2015-04-16 06:38:15.0121 2015-04-16 06:38:19.1703 4.1582171917  0 TH.GRZ1..BHE
    2015-04-16 06:38:11.9014 2015-04-16 06:38:34.4383 22.5369031429  0 None
    phase: 2017-04-29 00:56:32.9685  0 SX.WIMM..BHZ    sbqqrmbj03ce   2017-04-29   00:56:23.3900 P        None False

Each line in the marker file represents a visual marker. There are three kinds
of markers: event markers, phase markers, and basic markers.

* basic and phase markers can be associated with a certain NET.STA.LOC.CHA name
* basic and phase markers can either match a time instant or a time span
* phase markers can be associated to an event marker

The rules how to parse these files are somewhat cumbersome. Use the library
functions :py:func:`pyrocko.marker.load_markers`
:py:func:`pyrocko.marker.save_markers`. The marker file format is subject to be
replaced by a YAML based variant.
