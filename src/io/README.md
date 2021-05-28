
# Pyrocko I/O Modules

Input and output of seismic traces.

This module provides a simple unified interface to load and save traces to a few different file formats.  The data model used for the `pyrocko.trace.Trace` objects in Pyrocko is most closely matched by the Mini-SEED file format.  However, a difference is, that Mini-SEED limits the length of the network, station, location, and channel codes to 2, 5, 2, and 3 characters, respectively.

## Seismic Waveform IO

============ =========================== ========= ======== ======
format       format identifier           load      save     note
============ =========================== ========= ======== ======
Mini-SEED    mseed                       yes       yes
SAC          sac                         yes       yes      [#f1]_
SEG Y rev1   segy                        some
SEISAN       seisan, seisan.l, seisan.b  yes                [#f2]_
KAN          kan                         yes                [#f3]_
YAFF         yaff                        yes       yes      [#f4]_
ASCII Table  text                                  yes      [#f5]_
GSE1         gse1                        some
GSE2         gse2                        some
DATACUBE     datacube                    yes
SUDS         suds                        some
CSS          css                         yes
============ =========================== ========= ======== ======

## Metadata IO

* `pyrocko.io.quakeml` parses QuakeML (https://quake.ethz.ch/quakeml/) into a pyrocko data model.
* `pyrocko.io.stationxml` represents a FDSN stationXML model.

.. rubric:: Notes

.. [#f1] For SAC files, the endianness is guessed. Additional header
    information is stored in the `Trace`'s ``meta`` attribute.
.. [#f2] Seisan waveform files can be in little (``seisan.l``) or big endian
    (``seisan.b``) format. ``seisan`` currently is an alias for ``seisan.l``.
.. [#f3] The KAN file format has only been seen once by the author, and support
    for it may be removed again.
.. [#f4] YAFF is an in-house, experimental file format, which should not be
    released into the wild.
.. [#f5] ASCII tables with two columns (time and amplitude) are output - meta
    information will be lost.
