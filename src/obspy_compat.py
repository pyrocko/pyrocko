def to_pyrocko_trace(obspy_trace):
    '''
    Convert ObsPy trace object to Pyrocko trace object.
    '''

    from pyrocko import trace

    return trace.Trace(
        str(obspy_trace.stats.network),
        str(obspy_trace.stats.station),
        str(obspy_trace.stats.location),
        str(obspy_trace.stats.channel),
        tmin=obspy_trace.stats.starttime.timestamp,
        tmax=obspy_trace.stats.endtime.timestamp,
        ydata=obspy_trace.data,
        deltat=obspy_trace.stats.delta)


def to_pyrocko_traces(obspy_stream):
    '''
    Convert ObsPy stream object to list of Pyrocko trace objects.
    '''
    return [to_pyrocko_trace(obspy_trace) for obspy_trace in obspy_stream]


def to_pyrocko_events(obspy_catalog):
    '''
    Convert ObsPy catalog object to list of Pyrocko event objects.
    '''

    events = []
    for obspy_event in obspy_catalog:
        for origin in obspy_event.origins:

            

            events.append(model.Event(
                name=obspy_event.resource_id + origin.resource_id,
                time=origin.time.timestamp,
                lat=origin.latitude,
                lon=origin.longitude,
                depth=origin.depth,
                region=origin.region))



    return events


def snuffle(obspy_stream, obspy_inventory, obspy_catalog):
    '''
    '''

    from pyrocko import trace
    events = to_pyrocko_events(catalog)
    stations = to_pyrocko_stations(obspy_inventory)

    print trace.snuffle(
        to_pyrocko_traces(obspy_stream),
        events=events,
        stations=stations,
        want_markers=True)


def install_obspy_hooks():
    '''Add some Pyrocko methods to ObsPy classes.'''

    import obspy
    obspy.Stream.snuffle = snuffle
