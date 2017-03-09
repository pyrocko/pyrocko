Earthquake catalog search
=========================

pyrocko provides easy catalog access via the :class:`pyrocko.catalog`
module.

Searching the GlobalCMT catalog
--------------------------------

This example demonstrates how to query the `GlobalCMT <http://www.globalcmt.org/>`_ [#f1]_ database for events which occurred in 2011 in northern Chile.

::

    from pyrocko import catalog
    from pyrocko import util
    from pyrocko import model


    tmin = util.str_to_time('2011-01-01 00:00:00')  # beginning time of query
    tmax = util.str_to_time('2011-12-31 23:59:59')

    # create an instance of the global CMT catalog
    global_cmt_catalog = catalog.GlobalCMT()

    # query the catalog
    events = global_cmt_catalog.get_events(
        time_range=(tmin, tmax),
        magmin=2.,
        latmin=-35.,
        latmax=-20.,
        lonmin=-76.,
        lonmax=-65.)

    print 'Downloaded %s events' % len(events)
    print 'The last one is %s' % events[-1]

    # dump events to catalog
    model.dump_events(events, 'northern_chile_events.pf')

    >>> Downloaded 53 events
    >>> The last one is --- !pf.Event
    >>> lat: -28.03
    >>> lon: -71.55
    >>> time: 2011-12-07 22:23:14.250000
    >>> name: 201112072223A
    >>> depth: 22800.0
    >>> magnitude: 6.106838394015895
    >>> region: NEAR COAST OF NORTHERN C
    >>> catalog: gCMT
    >>> moment_tensor: !pf.MomentTensor
    >>>   mnn: 1.16e+17
    >>>   mee: -1.24e+18
    >>>   mdd: 1.1200000000000001e+18
    >>>   mne: 1.29e+17
    >>>   mnd: 1.61e+17
    >>>   med: 1.0900000000000001e+18
    >>>   strike1: 16.540029329929244
    >>>   dip1: 24.774772153067424
    >>>   rake1: 109.14904335232158
    >>>   strike2: 175.61123518070136
    >>>   dip2: 66.6800337700307
    >>>   rake2: 81.39111828783355
    >>>   moment: 1.622772319211786e+18
    >>>   magnitude: 6.106838394015895
    >>> duration: 5.4


Search for an event in Geofon catalog
--------------------------------------------------

Search for an event name only in the `Geofon <http://geofon.gfz-potsdam.de>`_ catalog [#f2]_ using :meth:`~pyrocko.catalog.Geofon`, with a given magnitude range and timeframe.

::

    from pyrocko import catalog, util

    tmin = util.ctimegm('2010-01-12 21:50:00')
    tmax = util.ctimegm('2010-01-13 03:17:00')  # ending time of query
    mag = 6.                                    # minimum magntiude (open end)

    # download event information from GEOFON web page

    geofon = catalog.Geofon()
    event_names = geofon.get_event_names(
        time_range=(tmin, tmax),
        magmin=mag)

    for event_name in event_names:
        event = geofon.get_event(event_name)
        print event

    >>> --- !pf.Event
    >>> lat: 18.37
    >>> lon: -72.55
    >>> time: 2010-01-12 21:53:11
    >>> name: gfz2010avtm
    >>> depth: 17000.0
    >>> magnitude: 7.2
    >>> region: Haiti Region
    >>> catalog: GEOFON


.. rubric:: Footnotes

.. [#f1] Dziewonski, A. M., T.-A. Chou and J. H. Woodhouse, Determination of earthquake source parameters from waveform data for studies of global and regional seismicity, J. Geophys. Res., 86, 2825-2852, 1981. doi:10.1029/JB086iB04p02825

.. [#f2] GEOFON Data Centre (1993): GEOFON Seismic Network. Deutsches GeoForschungsZentrum GFZ. Other/Seismic Network. doi:10.14470/TR560404. 
