# Pyrocko
### _A seismology toolkit for Python_
[![Build Status](https://travis-ci.org/pyrocko/pyrocko.svg?branch=master)](https://travis-ci.org/pyrocko/pyrocko) [![Coverage Status](https://coveralls.io/repos/github/pyrocko/pyrocko/badge.svg)](https://coveralls.io/github/pyrocko/pyrocko)

For documentation and installation instructions please see 
[http://pyrocko.org/](http://pyrocko.org/).

# Python 3 Support

## TODO

- [x] Include PyAVL
- [ ] Travis Py2/3 tests
- [ ] Create dedicated folders
  - [x] io
  - [ ] gui
  - [x] datasets
  - [ ] models
- [ ] Finish and Tag
- [ ] Build new docs (together w deplog mechanism?)
- [ ] Deployment mechanisms
  - [ ] PIP
  - [ ] Anaconda

## Module Progress

- [x] ``__init__.py``
- [x] ``ahfullgreen.py``
- [x] ``automap.py``
- [x] ``autopick.py``
- [x] ``beachball.py``
- [x] ``cake_plot.py``
- [x] ``cake.py``
- [x] ``catalog.py``
- [x] ``config.py``
- [x] ``crust2x2_data.py``
- [x] ``crust2x2.py``
- [x] ``crustdb_abbr.py``
- [x] ``crustdb.py``
- [x] ``css.py``
- [x] ``datacube.py``
- [x] ``dummy_progressbar.py``
- [x] ``edl.py`` *partial, not tested*
- [x] ``evalresp_ext.c``
- [x] ``evalresp.py``
- [x] ``eventdata.py``
- [x] ``ext/``
  - [x] ``ahfullgreen_ext.c``
  - [x] ``autopick_ext.c``
  - [x] ``datacube_ext.c``
  - [x] ``ims_ext.c``
  - [x] ``mseed_ext.c``
  - [x] ``orthodrome_ext.c``
  - [x] ``parstack_ext.c``
  - [x] ``signal_ext.c``
  - [x] ``util_ext.c``
- [x] ``fdsn/``
  - [x] ``__init__.py``
  - [x] ``enhanced_sacpz.py``
  - [x] ``resp.py``
  - [x] ``station.py``
  - [x] ``ws.py``
- [x] ``file.py``
- [x] ``fomosto/``
  - [x] ``__init__.py``
  - [x] ``ahfullgreen.py``
  - [x] ``dummy.py`` *not tested*
  - [x] ``poel.py`` *not tested*
  - [x] ``psgrn_pscmp.py``
  - [x] ``qseis2d.py``
  - [x] ``qseis.py``
  - [x] ``qssp.py``
- [x] ``fomosto_report/``
- [x] ``gcf.py``
- [x] ``geonames.py``
- [ ] ``gf/``
  - [x] ``__init__.py``
  - [x] ``builder.py``
  - [x] ``meta.py``
  - [x] ``seismosizer.py``
  - [x] ``server.py``
  - [x] ``store.py``
  - [x] ``targets.py``
  - [x] ``ws.py``
- [x] ``gmtpy.py``
- [x] ``gse1.py``
- [x] ``gse2_io_wrap.py``
- [x] ``gshhg.py``
- [x] ``gui_util.py``
- [x] ``guts_array.py``
- [x] ``guts.py``
- [x] ``hamster_pile.py`` *not tested*
- [x] ``hudson.py``
- [x] ``ims.py``
- [x] ``io_common.py``
- [x] ``io.py``
- [x] ``iris_ws.py``
- [x] ``kan.py``
- [x] ``marker_editor.py``
- [x] ``marker.py``
- [x] ``model.py``
- [x] ``moment_tensor.py``
- [x] ``moment_tensor_viewer.py``
- [x] ``mseed.py``
- [x] ``orthodrome.py``
- [x] ``parimap.py``
- [x] ``parstack.py``
- [x] ``pchain.py``
- [x] ``pile.py``
- [x] ``pile_viewer.py``
- [x] ``plot.py``
- [x] ``pz.py``
- [x] ``quakeml.py``
- [x] ``rdseed.py``
- [x] ``response_plot.py``
- [x] ``sac.py``
- [x] ``seed.py``
- [x] ``segy.py``
- [x] ``seisan_response.py``
- [x] ``seisan_waveform.py``
- [x] ``serial_hamster.py``
- [x] ``shadow_pile.py``
- [x] ``slink.py``
- [x] ``snuffler.py`` *WIP*
- [x] ``snuffling.py``
- [x] ``snufflings/``
- [x] ``spit.py``
- [x] ``suds.py``
- [x] ``tectonics.py``
- [x] ``topo/``
  - [x] ``__init__.py``
  - [x] ``dataset.py``
  - [x] ``etopo1.py``
  - [x] ``srtmgl3.py``
  - [x] ``tile.py``
- [x] ``trace.py``
- [x] ``util.py``
- [x] ``weeding.py``
- [x] ``yaff.py``

## Citation
Recommended citation for Pyrocko

> Heimann, Sebastian; Kriegerowski, Marius; Isken, Marius; Cesca, Simone; Daout, Simon; Grigoli, Francesco; Juretzek, Carina; Megies, Tobias; Nooshiri, Nima; Steinberg, Andreas; Sudhaus, Henriette; Vasyura-Bathke, Hannes; Willey, Timothy; Dahm, Torsten (2017): Pyrocko - An open-source seismology toolbox and library. V. 0.3. GFZ Data Services. http://doi.org/10.5880/GFZ.2.1.2017.001

[![DOI](https://img.shields.io/badge/DOI-10.5880%2FGFZ.2.1.2017.001-blue.svg)](http://doi.org/10.5880/GFZ.2.1.2017.001)

## License 
GNU General Public License, Version 3, 29 June 2007

Copyright © 2017 Helmholtz Centre Potsdam GFZ German Research Centre for Geosciences, Potsdam, Germany

Pyrocko is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
Pyrocko is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

## Contact
* Sebastian Heimann; 
  sebastian.heimann@gfz-potsdam.de

* Marius Isken; 
  marius.isken@gfz-potsdam.de

* Marius Kriegerowski; 
  marius.kriegerowski@gfz-potsdam.de 

```
Helmholtz Centre Potsdam German Research Centre for Geoscienes GFZ
Section 2.1: Physics of Earthquakes and Volcanoes
Helmholtzstraße 6/7
14467 Potsdam, Germany
```
