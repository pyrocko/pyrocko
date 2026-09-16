Earthquake catalog
==================

Pyrocko provides access to some online earthquake catalogs via the
:mod:`pyrocko.client.catalog` module. The recommended way to query them is
through :py:meth:`~pyrocko.squirrel.base.Squirrel.add_catalog`, which wraps
the same clients but integrates the results into Squirrel's unified,
cached data access.


QuakeML import
--------------

This example shows how to read `QuakeML <https://quake.ethz.ch/quakeml/docs/REC?action=AttachFile&do=get&target=QuakeML-BED-20130214b.pdf>` event catalogs using :func:`~pyrocko.io.quakeml.QuakeML.load_xml`.
The function :meth:`~pyrocko.io.quakeml.QuakeML.get_pyrocko_events()` is used to obtain events in pyrocko format.
If a moment tensor is provided as [``Mrr, Mtt, Mpp, Mrt, Mrp, Mtp``], this is converted to [``mnn, mee, mdd, mne, mnd, med``]. The strike, dip and rake values appearing in the pyrocko event are calculated from the moment tensor.

.. literalinclude :: /../../examples/readnwrite_quakml.py
    :language: python


Creating QuakeML from scratch
-----------------------------

.. literalinclude :: /../../examples/make_quakeml.py
    :language: python


Searching an online earthquake catalog
---------------------------------------

This example demonstrates how to query the `GlobalCMT
<http://www.globalcmt.org/>`_ [#f1]_ database for events which occurred in 2011
in northern Chile, using :py:meth:`~pyrocko.squirrel.base.Squirrel.add_catalog`
to declare it as a data source. Query arguments common to all queries made to
the catalog (here: the region and magnitude constraints) are passed via
``query_args``, as strings. Calling
:py:meth:`~pyrocko.squirrel.base.Squirrel.update` for the time span of
interest triggers the actual query, if the local copy of the catalog isn't
already up to date for it.

.. literalinclude :: /../../examples/squirrel_catalog_search.py
    :language: python

Download :download:`squirrel_catalog_search.py </../../examples/squirrel_catalog_search.py>`


We expect to see the following output:

::

    Downloaded 53 events
    The last one is
    --- !pf.Event
    lat: -28.03
    lon: -71.55
    time: '2011-12-07 22:23:14.25'
    depth: 22800.0
    name: '201112072223A'
    magnitude: 6.106838394015895
    region: 'NEAR COAST OF NORTHERN C'
    catalog: 'gCMT'
    moment_tensor: !pf.MomentTensor
      mnn: 1.16e+17
      mee: -1.24e+18
      mdd: 1.1200000000000001e+18
      mne: 1.29e+17
      mnd: 1.61e+17
      med: 1.0900000000000001e+18
      strike1: 16.540029329929244
      dip1: 24.774772153067424
      rake1: 109.14904335232158
      strike2: 175.61123518070136
      dip2: 66.6800337700307
      rake2: 81.39111828783355
      moment: 1.622772319211786e+18
      magnitude: 6.106838394015895
    duration: 5.4

The same pattern works for the other catalogs supported by
:py:meth:`~pyrocko.squirrel.base.Squirrel.add_catalog` (currently GEOFON and
ISC) - just swap the catalog name and ``query_args``, e.g.
``sq.add_catalog('geofon', query_args=dict(magmin='6.'))``.

For a one-off query where Squirrel's indexing/caching would be unwanted
overhead, the underlying clients in :mod:`pyrocko.client.catalog` can be
used directly:

::

    from pyrocko.client.catalog import GlobalCMT

    events = GlobalCMT().get_events(
        time_range=(tmin, tmax), magmin=2.,
        latmin=-35., latmax=-20., lonmin=-76., lonmax=-65.)


.. rubric:: Footnotes

.. [#f1] Dziewonski, A. M., T.-A. Chou and J. H. Woodhouse, Determination of earthquake source parameters from waveform data for studies of global and regional seismicity, J. Geophys. Res., 86, 2825-2852, 1981. doi:10.1029/JB086iB04p02825
