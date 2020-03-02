# Changelog

All notable changes to Pyrocko will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- `fomosto`:
  - Support QSSP 2017
  - Support QSSP PPEG variant
  - Add subcommand `tttlsd` to fill holes in travel time tables. Uses eikonal
    solver to fill the holes.
  - Allow setting receiver depth for `tttview`.
- `gf`:
  - Support ``velocity`` and ``acceleration`` as stored GF quantities.
  - Support squared half-sinusoid source time function.

### Changed
- In `trace.transfer`, bypass FFTs for flat responses.

### Fixed
- Fix problems with corrupt channel info text tables from FDSN.
- Correct reading of SEGY files with IEEE floating point values.
- Correct query parameters for ISC catalog, previous versions where querying
  for HH:HH:SS instead of HH:MM:SS.
- Fix scenario generator crashes.
- Fix incorrect error handling in GF waveform synthesis
  (`store_calc_timeseries`).

## [v2020.02.10] - 2020-02-10

### Added
- Support querying all stations with `GLOBAL` from ISC.
- Support event queries in `client.fdsn`.
- Catalog snuffling: support FDSN event queries.
- Add `trace_scale` setting in Snuffler configuration file.
- Support 1-2-3 as a valid channel triplet.
- Support `elastic2` component scheme in Fomosto QSEIS backend (pure explosion
  sources).
- Support Burger's elasticity parameters in `cake` and `gf.psgrn_pscmp`.
- Include updated old GmtPy tutorial it documentation.
- Snuffler: can now use `<tab>` and `<shift>-<tab>` to iterate through phase
  markers of active event.
- New dataset: Pleistocene and Holocene volcano database from Smithsonian
  Institution.

### Changed
- All location-based objects like events and stations should now fully support
  Cartesian offset coordinates.
- Scenario: interface, behaviour and defaults improved.
- Snuffler: trace to station lookup now supports both: `(net, sta)` and `(net,
  sta, loc)`. The latter more specific one has precedence.
- Automap: changed appearance of GNSS velocities.

### Fixed
- Fix a bug affecting origin time and STF of `gf.RectangularSource`.
- Fix conversion of QuakeML phase lacking phase hint to Pyrocko marker.
- Fix ignored `timeout` argument in `client.fdsn.dataselect`.
- Fix error in polygon-on-sphere cutting (`beachball` module).
- Fix errors in point in polygon-on-sphere checks.
- Fix various errors in GSHHG data dry/wet area masking.
- Fix crashes with bad StationXML files, improve robustness.
- Fix problems with crazy GPS infos in Datacube recordings. Time interpolation
  is now much more robust.
- Fix crashes in recursive STA/LTA.
- Fix a numerical bug in recursive STA/LTA.
- Improved SeedLink acquisition.

## [v2019.06.06] - 2019-06-06

### Added
- Shortcut to get Kite `Scene` from `SatelliteResult` and direct
  `KiteSceneTarget`.
- Improve documentation of `pyrocko.gf` with a special topic chapter.
- Combined volume change plus CLVD parameterized source: `gf.VLVDSource`.
- Method to perturb `cake` earth models.
- Support cross-section views in beachball plotting.

### Changed
- Automap now defaults to not show any scale.

## [v2019.05.03] - 2019-05-03

### Fixed
- Improved compatibility with Anaconda.

## [v2019.05.02] - 2019-05-02

### Added
- `gf`:
  - CombiSource, which can handle a set of different sources as a single source
    object
  - resonator stf
- `guts`:
  - Function to get Guts attributes.
  - `ypath` functions to set Guts attributes via pattern.
- Snuffler: `clip_traces` configurable in configuration file.
- New client: ISC catalog interface.
- Add a function to create phase markers from QuakeML.
- Can now use Gutenberg Richter magnitude distribution when creating synthetic
  scenario events.
- Support for handling of sparse gnss components.

### Changed
- gf:
  - Now the discretised point sources have their true time set, time shifting
    is done only before and after c function calls.
  - Source times are handled in double precision.
  - Static store requests fallback to zero time to handle absolute source
    times.
  - Store ext gives individual index for out of bounds error for targets.
- Geofon catalog web format changed, now parsing GeoJSON instead of HTML.
- Instrument response deconvolution gives more explicit warnings about
  inconsistencies.
- Changed default of `model.gnss` correlation factor to 0.0.
- Add flag to dump progressbar instead of showing the progressbar.
- Can now handle and plot GNSS stations lacking complete set of component
  orientations.
- Event depth is now optional (model).

### Fixed
- gf:
  - low level errors in time handling
  - ExplosionSource for non-volume sources
- `obspy_compat` event conversion
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

## [v2019.05.02] - 2019-05-02

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


## [v2018.09.13] - 2018-09-13

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


## [v2018.01.29] - 2018-01-29

### Added
- `pyrocko.obspy_compat` compatibility with basic ObsPy classes (see https://pyrocko.org/docs/)
- Added examples for ObsPy compatability, StationXML and QuakeML
- Added `Source.outline`

### Changed


### Fixed
- `geofon` updated page parser
- old `requests` compatibility
- various bugfixes

## [v2018.01.16] - 2018-01-16

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

## [v2017.12.13] - 2017-12-13

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

## [v2017.11.22] - 2017-11-22

### Added
- QuakeML - Include moment tensor to `pyrocko.model.event`

### Changed
- deployment polishing to `pip` and Anaconda
- documentation

### Fixed
- `pyrocko.pile` caching
- automap SRTM download authentication

## [v2017.11.16] - 2017-11-16

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
