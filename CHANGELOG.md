# Changelog

All notable changes to Pyrocko will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [unreleased]

### Added
- `fdsn` adding event query
- `snuffling.catalog` adding FDSN event query
- Snuffler adding `trace_scale` to configuration
- Added Burger elasticity parameters to cake and `gf.psgrn_pscmp`
- Added Smithsonian Volcano database to `pyrocko.dataset`
- Added Tile export to 3D printable formats

### Fixed
- `seismosizer` STF timing of `RectangularSource`s.

## [v2019.05.03] 2019-05-03

### Added
- Geographical views for beachball plotting.

### Fixed
- Improved compatibility with Anaconda.

## [v2019.05.02] 2019-05-02

### Added
- gf:
  - CombiSource, which can handle a set of different sources as a single source object
  - resonator stf
- guts:
  - Function to get guts attributes
  - ypath functions
- snuffler: clip_traces configurable by config
- client: ISC catalog interface
- quakeml phase marker extraction
- random draw of magnitude from gutenberg richter distribution and use of it in scenario
- handling of sparse gnss components

### Changed
- gf:
  - Now the discretized point sources have their true time set, time shifting is done only before and afther c function calls.
  - Source times are handled on double precision
  - Static store requests fallback to zero time to handle absolute source times
  - Store ext gives indivudal index out of bounds error for targets
- Geofon catalog web format changed, parse geojson instead of html
- Instrument response deconvolutions gives more explicit warnings about inconsistencies
- model.gnss correlation factor default of 0.
- Added flag to dump progressbar instead of showing the progressbar
- Can now handle and plot GNSS stations lacking complete set of component orientations.
- Event depth is now optional (model)

### Fixed
- gf:
  - low level errors in time handling
  - ExplosionSource for non-volume sources
- obspy_compat event conversion
- Problem with numpy scalars in source and target objects
- Conversion of quakeml piks to pyrocko phasepiks use phase polarities strings
- snuffling broken load dialogs (QT5)
- trace: deltat rounding error function
- segy functions
- model.gnss covar
- scenario: perfomance improvement
- geofon catalog: problem with multiple solutions for some events
- setup: UnicodeDecodeError
- fuzzy beachball plotting bugs

## [v2019.05.02] 2019-05-02

### Added

- Anaconda builds for python 3.6 and 3.7
- Green's Mill
- gf: rectangular source check for point source discretisation
- stationxml: using a flat response function is enabled
- guts: support for time stamps in local time
- datasets/geonames: get city by name
- gf: Eikonal solver
- plot: fuzzy beachballs
- trace: has a function to fix rounding errord

### Changed
- gf: improved summation of static gf components
- FDSN: configurable time out for requests

### Fixed
- rdseed get_station fix
- gf: quantity velocity from displacement through `num.diff`


## [v2018.09.13] 2019-9-13

### Added
- Snuffler map added
  - Bing aerial images
  - Distance and area measure
  - Toggle lines and cities
- GF: Server added JSON API
- Snuffler: and Jackseis now accept YAML event files
- Snuffler: polarity picking in
- fomosto: ttt now checks for holes in travel-time tables
- Guts:
  - can now specify output yaml style for strings
  - improved XML namespace support
  - add utility module to help with guts structure updaters
- model: GNSS displacement components can have covariances


### Changed
- gf:
  - improved handling of derived magnitude sources
  - slip-driven rectangular source, explosion

### Fixed
- Fixed Bug in `ExplosionSource` (incorrect moment conversion)
- Fixed bug in polygon coordinates
- Fixed SSL issues
- Fixed SRTM tile download
- Fixed `fomosto tttextract --output`
- Various minor bugfixes
- Various compatibility fixes
- Updated web service URLs and online catalog parsers


## [v2018.01.29] 2018-01-29

### Added
- `pyrocko.obspy_compat` compatibility with basic ObsPy classes (see https://pyrocko.org/docs/)
- Added examples for ObsPy compatability, StationXML and QuakeML
- Added `Source.outline`

### Changed


### Fixed
- `geofon` updated page parser
- old `requests` compatibility
- various bugfixes

## [v2018.01.16] 2018-01-16

### Added
- improvements to `colosseo` / `scenario`
- `model.load_events()` now accepts `yaml` based event files

### Changed
- changes on geofon event web pages require an update in `catalog.geofon`
- better handling of defaults in `guts`, guts-derived default objects are now automatically cloned
- follow changed SRTM url

### Fixed
- error in string/time conversion (affecting date formats with more than one '.')
- py2/3 related bugs in `cake` and `guts`
- incorrect handling of moment/magnitude in `moment_tensor.MomentTensor.randam_mt()`

## [v2017.12.13] 2017-12-13

### Added
- new `pyrocko.model`: `GNSSCampaign` model
- new `pyrocko.gf`:  `GNSSCampaignTarget`
- New module `pyrocko.scenario`: Create earthquake data scenarios on the fly, outputs are
  - Waveforms model
  - InSAR deformation model (with `kite`)
  - GNSS data
 Use `colosseo` CLI tool to initialize and calculate your earthquake scenarios.

### Changed
- `pyrocko.pile`: caching
- `snuffler`: improved OpenStreetmap visual appearance and logging
- `snuffler`: marker table print menu
- `snuffler.maps`: upgrade to OpenLayers4

## [v2017.11.22] 2017-11-22

### Added
- QuakeML - Include moment tensor to `pyrocko.model.event`

### Changed
- deployment polishing to `pip` and Anaconda
- documentation

### Fixed
- `pyrocko.pile` caching
- automap SRTM download authentication

## [v2017.11.16] 2017-11-16

### Added
- Python 2/3 support
- Community support chat
- new deployment for Anaconda and `pip`
- Installation through python pip and Anaconda
- Built-in Snufflings added: Maps, Seismosizer and Cake

### Changed
- Version naming has changed to rolling release style  (e.g. v2017.11.16)
- Reorganized module hierarchy (backwards-compatible)
- documentation and examples
