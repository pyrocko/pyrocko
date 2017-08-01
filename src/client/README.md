# Pyrocko Client Submodule

The pyrocko software package comes with several clients for requesting online earthquake catalogs and waveform archives.

## Waveform Access

* `pyrocko.client.iris` gives access to waveform from IRIS (http://service.iris.edu/).
* `pyrocko.client.fdsn` is a general client for providing an FDSN web service implementation (http://www.fdsn.org/).

## Earthquake Catalogs

* `pyrocko.client.geofon` Access the Geofon earthquake catalog (http://geofon.gfz-potsdam.de/)
* `pyrocko.client.usgs` Query the USGS earthquake catalog (https://earthquake.usgs.gov/)
* `pyrocko.client.globalcmt` Get earhquake events from the Global CMT catalog (http://www.globalcmt.org/)
* `pyrocko.client.kinherd` Kinherd's earthquake catalog at (http://kinherd.org/quakes/)
* `pyrocko.client.saxony` Regional Catalog of Saxony, Germany from the University of Leipzig (http://home.uni-leipzig.de/collm/auswertung_temp.html)

(Also accessible through `pyrocko.client.catalog`)
