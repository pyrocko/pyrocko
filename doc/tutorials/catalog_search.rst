Earthquake catalog search
=========================


Search for an event name
--------------------------------------------------
Search for an event name only in the geofon catalog using :py:meth:`~pyrocko.catalog.Geofon`, with a given magnitude and timeframe.
::

    
    from pyrocko import catalog, util
    
    
    tmin = util.ctimegm('2010-01-12 21:50:00')  # beginning time of query
    tmax = util.ctimegm('2010-01-13 03:17:00') # ending time of query 
    mag=6. #minimum magntiude (open end)
    
    
    
    # download event information from GCMT web page
    
    geofon = catalog.GlobalCMT()
    event_names = geofon.get_event_names(
        time_range=(tmin, tmax),
        magmin=mag)
    
    #This puts out a list of events in the timeframe
    print event_names

    
    
Search for an event name and print out event information
---------------------------------------------------------
Search for an event name only in the geofon catalog using :py:meth:`~pyrocko.catalog.Geofon`, with a given magnitude and timeframe.
::


    from pyrocko import catalog, util
    
    
    tmin = util.ctimegm('2010-01-12 21:50:00')  # beginning time of query
    tmax = util.ctimegm('2010-01-13 03:17:00') # ending time of query 
    mag=6. #minimum magntiude (open end)
    
    
    # download event information from GEOFON web page
    
    geofon = catalog.Geofon()
    event_names = geofon.get_event_names(
        time_range=(tmin, tmax),
        magmin=mag)
    
    
    for event_name in event_names:
    
        event = geofon.get_event(event_name)
        
        print event 