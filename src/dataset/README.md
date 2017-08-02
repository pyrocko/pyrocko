# Pyrocko Dataset Submodule

The `pyrocko.dataset` submodule gives access to several helpful geological and geophysical datasets:

# Seismic Velocity Datasets
* `pyrocko.dataset.crust2x2` Interface to CRUST2.0 model by Laske, Masters and Reif (https://igppweb.ucsd.edu/~gabi/crust2.html).
* `pyrocko.dataset.crustdb` Accesses the Global Crustal Database (https://earthquake.usgs.gov/data/crust) for statistical analysis of empirical velocity profiles. [Mooney et al., 1998]

# Topographic Datasets
* `pyrocko.dataset.topo.etopo1` Download and convinient access to ETOPO1, a 1 arc-minute global relief model (https://www.ngdc.noaa.gov/mgg/global/).
* `pyrocko.dataset.topo.srtmgl3` Download and access gridded relief data from SRTMGL3 (https://lpdaac.usgs.gov/dataset_discovery/measures/measures_products_table/srtmgl3_v003).

# Tectonic Datasets
* `pyrocko.dataset.tectonics` module consists of the tectonic datasets:
  * `pyrocko.dataset.PeterBird2002` An updated digital model of plate boundaries. (Geochemistry, Geophysics, Geosystems 4.3, 2003)
  * `pyrocko.dataset.GSRM1` An integrated global model of present-day plate motions and plate boundary deformation (Kreemer, C., W.E. Holt, and A.J. Haines, 2003)

# Geographical Datasets
* `pyrocko.dataset.gshhg` An interface to the Global Self-consistent Hierarchical High-resolutions Geography Database (GSHHG; http://www.soest.hawaii.edu/wessel/gshhg/).
* `pyrocko.dataset.geonames` Providing city names and population size from http:/geonames.org.
