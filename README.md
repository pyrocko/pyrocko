# Pyrocko
### _A seismology toolkit for Python_

[![Build Status](https://drone.pyrocko.org/api/badges/pyrocko/pyrocko/status.svg?ref=refs/heads/master)](https://drone.pyrocko.org/pyrocko/pyrocko)
[![Anaconda-Server Badge](https://anaconda.org/pyrocko/pyrocko/badges/version.svg)](https://conda.anaconda.org/pyrocko)
[![PyPI](https://img.shields.io/pypi/v/pyrocko.svg)](https://pypi.python.org/pypi/pyrocko/)


## Installation

Pyrocko can be installed on various operating systems and in many different
installation styles. Please consult the [Pyrocko Installation Manual](https://pyrocko.org/docs/current/install/) for details.

### System wide installation from source

```
git clone https://git.pyrocko.org/pyrocko/pyrocko.git
cd pyrocko
python install.py deps system
python install.py system
```

### User installation from source

```
git clone https://git.pyrocko.org/pyrocko/pyrocko.git
cd pyrocko
pip install .  # only install into isolated environments like this!
```

### Installation with Anaconda

Anaconda3 packages are available for Linux, OSX and Windows ([details](https://pyrocko.org/docs/current/install/packages/anaconda.html)).

```
conda install -c pyrocko pyrocko
```

### User installation with Python pip

Binary pip packages are available for Linux and Windows ([details](https://pyrocko.org/docs/current/install/packages/pip.html)).

```
pip install --user pyrocko
pip install --user --only-binary :all: PyQt5
```

## Documentation

Documentation and usage examples are available online at https://pyrocko.org/docs/current

## Community Support

Community support at [https://hive.pyrocko.org](https://hive.pyrocko.org/signup_user_complete/?id=9edryhxeptdbmxrecbwy3zg49y).

## Citation
The recommended citation for Pyrocko is: (You can find the BibTeX snippet in the
[`CITATION` file](CITATION.bib)):

> Heimann, Sebastian; Kriegerowski, Marius; Isken, Marius; Cesca, Simone; Daout, Simon; Grigoli, Francesco; Juretzek, Carina; Megies, Tobias; Nooshiri, Nima; Steinberg, Andreas; Sudhaus, Henriette; Vasyura-Bathke, Hannes; Willey, Timothy; Dahm, Torsten (2017): Pyrocko - An open-source seismology toolbox and library. V. 0.3. GFZ Data Services. https://doi.org/10.5880/GFZ.2.1.2017.001

[![DOI](https://img.shields.io/badge/DOI-10.5880%2FGFZ.2.1.2017.001-blue.svg)](https://doi.org/10.5880/GFZ.2.1.2017.001)

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
