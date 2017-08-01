# Pyrocko Model Submodule

The `pyrocko.model` submodule holds abstract data model of common geophysical data structures

* `pyrocko.model.Event` represents an earthquake event.
* `pyrocko.model.Trace` is a single seismic trace which can be filtered, snuffled and otherwise filtered
* `pyrocko.model.Station` represents a seismic station with location and network/name identifiers.

* The module `pyrocko.model.quakeml` parses QuakeML (https://quake.ethz.ch/quakeml/) into a pyrocko data model.
* The module `pyrocko.model.fdsn_station` represents a FDSN station model.
