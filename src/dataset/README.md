

# Pyrocko Dataset Submodule

The `pyrocko.dataset` submodule gives access to several helpful geo datasets:

## Seismic Velocity Datasets

* `pyrocko.dataset.crust2x2` Interface to CRUST2.0 global seismic velocity model (https://igppweb.ucsd.edu/~gabi/crust2.html) [Bassin et al., 2000].
* `pyrocko.dataset.crustdb` Accesses the Global Crustal Database (https://earthquake.usgs.gov/data/crust) delivering empirical velocity measurements of the earth for statistical analysis. [Mooney et al., 1998]

## Topographic Datasets Submodule

* `pyrocko.dataset.topo` Access to topograhic datasets in different resolutions.

## Tectonic Datasets `pyrocko.dataset.tectonics`

* `pyrocko.dataset.tectonics.PeterBird2003` An updated digital model of plate boundaries. (http://peterbird.name/publications/2003_PB2002/2003_PB2002.htm) [P. Bird, 2003]
* `pyrocko.dataset.tectonics.GSRM1` An integrated global model of present-day plate motions and plate boundary deformation [Kreemer, C., W.E. Holt, and A.J. Haines, 2003]

## Geographic Datasets

* `pyrocko.dataset.gshhg` An interface to the Global Self-consistent Hierarchical High-resolutions Geography Database (GSHHG; http://www.soest.hawaii.edu/wessel/gshhg/) [Wessel et al, 1998].
* `pyrocko.dataset.geonames` Providing city names and population size from http://geonames.org.
