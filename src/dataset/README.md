# Pyrocko Dataset Submodule

The `pyrocko.dataset` submodule gives access to several helpful geological and geophysical datasets:

# Seismic Velocity Datasets
* `pyrocko.dataset.crust2x2` Interface to CRUST2.0 model by Laske, Masters and Reif (https://igppweb.ucsd.edu/~gabi/crust2.html).
* `pyrocko.dataset.crustdb` Accesses the Global Crustal Database (https://earthquake.usgs.gov/data/crust) for statistical analysis of empirical velocity profiles. [Mooney et al., 1998]

# Topographic Datasets
* `pyrocko.dataset.topo` Access to topograhic datasets in different resolutions.

# Tectonic Datasets
* `pyrocko.dataset.tectonics` module offering tectonic data:
* `pyrocko.dataset.tectonics.PeterBird2003` An updated digital model of plate boundaries. (Geochemistry, Geophysics, Geosystems 4.3, 2003)
* `pyrocko.dataset.tectonics.GSRM1` An integrated global model of present-day plate motions and plate boundary deformation (Kreemer, C., W.E. Holt, and A.J. Haines, 2003)

# Geographic Datasets
* `pyrocko.dataset.gshhg` An interface to the Global Self-consistent Hierarchical High-resolutions Geography Database (GSHHG; http://www.soest.hawaii.edu/wessel/gshhg/).
* `pyrocko.dataset.geonames` Providing city names and population size from http:/geonames.org.
