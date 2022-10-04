# Pyrocko client subpackage

The Pyrocko software package comes with several clients for requesting online earthquake catalogs and waveform archives.

## Waveform access

* `pyrocko.client.fdsn` Is a client for FDSN web services (http://www.fdsn.org/).
* `pyrocko.client.iris` Gives access to waveform data from the IRIS archive (http://service.iris.edu/).

## Earthquake catalogs

* `pyrocko.client.catalog.Geofon` Access the Geofon earthquake catalog (http://geofon.gfz-potsdam.de/)
* `pyrocko.client.catalog.GlobalCMT` Get earthquakes from the Global CMT catalog (http://www.globalcmt.org/)
* `pyrocko.client.catalog.USGS` Query the USGS earthquake catalog (https://earthquake.usgs.gov/)
* `pyrocko.client.catalog.Saxony` Regional Catalog of Saxony, Germany from the University of Leipzig (http://home.uni-leipzig.de/collm/auswertung_temp.html)

(Also accessible through `pyrocko.client.catalog`)
