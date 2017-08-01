# Pyrocko Models Submodule

The `pyrocko.models` submodule holds abstract data models of common geophysical data structures

* `pyrocko.models.Event` represents an earthquake event.
* `pyrocko.models.Trace` is a single seismic trace which can be filtered, snuffled and otherwise filtered
* `pyrocko.models.Station` represents a seismic station with location and network/name identifiers.

* The module `pyrocko.models.quakeml` parses QuakeML (https://quake.ethz.ch/quakeml/) into a pyrocko data model.
* The module `pyrocko.models.fdsn_station` represents a FDSN station model.
