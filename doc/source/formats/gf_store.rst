Pyrocko GF store file format
----------------------------

Sebastian Heimann


Abstract
........

This document describes the storage file formats used for the Green's function 
(GF) databases in the Pyrocko-GF framework [1]_.


Overview
........

A Pyrocko-GF store is an ordinary file system directory with a specific interal
structure. It must contain one mandatory file and may contain several optional
files and sub-directories with specific names (section
:ref:`gf-store-layout`). The mandatory file must be named ``config`` and
contains the GF store's meta information in the `YAML
<https://en.wikipedia.org/wiki/YAML>`_ format (section :ref:`gf-store-config`).
Any non-empty GF store additionally contains two platform-independent binary
files: a file named ``index``  and a file named ``traces`` (section
:ref:`gf-store-binary-files`).

The binary files ``index`` and ``traces`` are designed to be simple.
Conceptually, both files together define the mapping of a scalar
integer number (the `low-level index`) to snippets of regularly sampled
time series (referred to as `traces`). Each trace is composed of a
floating point array of GF samples, the time instant of its first sample and
the number of elements in the array. All traces in the GF store share the
same sampling rate.

A discrete, truncated, and sparse version of the GF is stored. Precisely, the
stored samples are

.. math:: 

    G[j,i] = G[j(x_s, x_r, k), i(t)], \qquad \textrm{with} \qquad  i \in [i_{\textrm{min}}[j], i_{\textrm{max}}[j]] \qquad \textrm{and} \qquad  j \in [0, N-1],

where :math:`j` is the combined integer low-level index, a mapping of source
coordinates :math:`x_s`, receiver coordinates :math:`x_r` and the GF component
:math:`k`, and :math:`i` is a mapping of the time :math:`t`. Values outside of
the time range of stored values are treated with repeating end-points, as

.. math::

    G[j, i<i_{\textrm{min}}[j]] = G[j, i_{\textrm{min}}[j]] \qquad \textrm{ and } \qquad G[j, i>i_{\textrm{max}}[j]] = G[j, i_{\textrm{max}}[j]].

The GF must be sampled at exact multiples of the sampling interval
:math:`\Delta t`, at times :math:`t_i = i \; \Delta t`. The instant 
:math:`t = 0` corresponds to the time of the impulse-like source excitation.

How physical coordinates and component number :math:`(x_s, x_r, k)` map into
the low-level index :math:`j` is specified in the meta information of the GF
store. Different mappings are available to support different source-receiver
geometries and to exploit problem symmetries.


.. _gf-store-layout:

Directory layout
................

The following table lists the entries of a GF store. Not all entries must be 
present.

+---------------+---------------+--------------------------------------+
|            **GF store directory contents**                           |
+---------------+---------------+--------------------------------------+
| *File name*   | *File type*   | *Description*                        |
+---------------+---------------+--------------------------------------+
| ``config``    | YAML file     | Store meta-information, extent and   |
|               |               | index mapping                        |
|               |               | (:ref:`gf-store-config`).            |
+---------------+---------------+--------------------------------------+
| ``index``     | binary file   | GF trace metadata                    |
|               |               | (:ref:`gf-store-index`).             |
+---------------+---------------+--------------------------------------+
| ``traces``    | binary file   | GF trace sample data                 |
|               |               | (:ref:`gf-store-traces`).            |
+---------------+---------------+--------------------------------------+
| ``decimated`` | sub-directory | Directory with decimated variants of |
|               |               | the store.                           |
+---------------+---------------+--------------------------------------+
| ``phases``    | sub-directory | Directory with travel-time           | 
|               |               | interpolation tables.                |
+---------------+---------------+--------------------------------------+
| ``extra``     | sub-directory | Directory with extra information.    |
|               |               | Used to to store backend-specific    |
|               |               | modelling input parameters.          |
+---------------+---------------+--------------------------------------+


.. _gf-store-config:

The ``config`` file
...................

The ``config`` file contains all meta information needed to correctly interpret
the stored Green's function. It may additionally contain information about the
modelling code, the earth model used, author information, citations, etc. Its
file format is the widely used YAML format, which provides a good compromise
between machine and human readability. It contains a common set of entries
available for all GF types plus extra entries for specific GF types.

See the Pyrocko reference manual for lists of available entries:

* Common entries: :py:class:`~pyrocko.gf.meta.Config`
* Type A store specific: :py:class:`~pyrocko.gf.meta.ConfigTypeA`
* Type B store specific: :py:class:`~pyrocko.gf.meta.ConfigTypeB`
* Type C store specific: :py:class:`~pyrocko.gf.meta.ConfigTypeC`


.. _gf-store-binary-files:

Binary files
............

All numbers are encoded in little endian format. Real numbers are encoded as
IEEE 754 32-bit or 64-bit floating-point values. 


.. _gf-store-index:

The ``index`` file
,,,,,,,,,,,,,,,,,,


The ``index`` file of the GF store is composed of a 12-byte header and a
sequence of 24-byte records, one record for each trace in the database. The
following tables define there internal structure.

+-----------------+-----------------------+--------------------------------------------------------------+
|                   **Index file header (12 bytes)**                                                     |
+-----------------+-----------------------+--------------------------------------------------------------+
| ``nrecords``    | `8-byte unsigned int` | Total number of records (traces) in the store :math:`N`.     | 
+-----------------+-----------------------+--------------------------------------------------------------+
| ``deltat``      | `4-byte float`        | Common sampling interval :math:`\Delta t`  of the GF traces. |
+-----------------+-----------------------+--------------------------------------------------------------+


+-----------------+-----------------------+--------------------------------------------------------------+
|                   **Index file record (24 bytes)**                                                     |
+-----------------+-----------------------+--------------------------------------------------------------+
| ``data_offset`` | `8-byte unsigned int` | Byte offset of first sample of the GF trace in the           |
|                 |                       | ``traces`` file. Special flag values are 0: trace is         |
|                 |                       | missing, 1: all samples in the trace are zero, 2: trace is   |
|                 |                       | short (one or two samples long).                             |
+-----------------+-----------------------+--------------------------------------------------------------+
| ``itmin``       | `4-byte signed int`   | Temporal onset of the GF trace in number of samples          |  
|                 |                       | :math:`i_{\textrm{min}}[j]`.                                 |
+-----------------+-----------------------+--------------------------------------------------------------+
| ``nsamples``    | `4-byte unsigned int` | Number of samples in the GF trace:                           |
|                 |                       | :math:`i_{\textrm{max}}[j] - i_{\textrm{min}}[j] + 1`.       |
+-----------------+-----------------------+--------------------------------------------------------------+
| ``begin_value`` | `4-byte float`        | Value the GF trace's first sample                            |
|                 |                       | :math:`G[j, i_{\textrm{min}}[j]]`, typically zero. This      |
|                 |                       | value is used when the GF trace has to be extrapolated to    |
|                 |                       | times :math:`t < t_{i_{\textrm{min}}[j]}`.                   |
+-----------------+-----------------------+--------------------------------------------------------------+
| ``end_value``   | `4-byte float`        | Value the GF trace's last sample                             |
|                 |                       | :math:`G[j, i_{\textrm{max}}[j]]`, typically zero, or the    |
|                 |                       | static offset -- This value is used when the GF trace has to |
|                 |                       | be extrapolated to times                                     |
|                 |                       | :math:`t < t_{i_{\textrm{min}}[j]}`.                         |
+-----------------+-----------------------+--------------------------------------------------------------+


.. _gf-store-traces:

The ``traces`` file
,,,,,,,,,,,,,,,,,,,

The ``traces`` file of the GF store is composed of 32 bytes of empty space at
the beginning, followed by zero or more trace data allocations (GF trace sample
arrays). Offset and length of each allocated trace are given in the index
record. First and last value of the trace data must match their counterparts in
the index record, ``begin_value`` and ``end_value``. This redundancy can be
used for integrity checks. No allocation is made for short traces, i.e. traces
with 2 samples, representing step functions.


.. [1] Sebastian Heimann, Hannes Vasyura-Bathke, Henriette Sudhaus, Marius
    Paul Isken, Marius Kriegerowski, Andreas Steinberg, and Torsten Dahm. "A
    Python framework for efficient use of pre-computed Green's functions in
    seismological and other physical forward and inverse source problems."
    Solid Earth 10, no. 6 (2019): 1921-1921.
