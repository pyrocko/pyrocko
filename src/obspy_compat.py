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
    if not obspy_catalog:
        return None

    from pyrocko import model

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


def to_pyrocko_stations(obspy_inventory):
    if not obspy_inventory:
        return None

    from pyrocko import model
    stations = []
    for net in obspy_inventory.networks:
        for sta in net.stations:
            stations.append(
                model.Station(
                    lat=sta.latitude,
                    lon=sta.longitude,
                    elevation=sta.elevation,
                    network=net.code,
                    station=sta.code,
                    location='',
                    channels=[
                        model.station.Channel(
                            name=cha.code,
                            azimuth=cha.azimuth,
                            dip=cha.dip) for cha in sta.channels]
                    ))

    return stations


def to_obspy_stream(pyrocko_pile):
    import obspy
    stream = obspy.Stream()
    stream.extend([to_obspy_trace(tr) for tr in pyrocko_pile.iter_all()])
    return stream


def to_obspy_trace(pyrocko_trace):
    '''Convert Pyrocko trace to ObsPy Trace
    '''
    import obspy

    obspy_trace = obspy.Trace(
        data=pyrocko_trace.ydata,
        header=obspy.core.trace.Stats(
            dict(
                network=pyrocko_trace.network,
                station=pyrocko_trace.station,
                location=pyrocko_trace.location,
                channel=pyrocko_trace.channel,
                delta=pyrocko_trace.deltat,
                starttime=pyrocko_trace.tmin,
                endtime=pyrocko_trace.tmax)
            ))

    return obspy_trace


def snuffle(obspy_stream, obspy_inventory=None, obspy_catalog=None):
    '''
    '''
    from pyrocko import trace
    import obspy
    events = to_pyrocko_events(obspy_catalog)
    stations = to_pyrocko_stations(obspy_inventory)

    if isinstance(obspy_stream, obspy.Trace):
        obspy_stream = obspy.Stream(traces=obspy_stream)

    trace.snuffle(
        to_pyrocko_traces(obspy_stream),
        events=events,
        stations=stations,
        want_markers=True)


def plant():
    '''Add some Pyrocko methods to ObsPy classes.'''

    import obspy
    obspy.Trace.to_pyrocko_trace = to_pyrocko_trace
    obspy.Trace.snuffle = snuffle

    obspy.Stream.snuffle = snuffle
    obspy.Stream.to_pyrocko_traces = to_pyrocko_traces

    obspy.core.event.Catalog.to_pyrocko_events = to_pyrocko_events
    obspy.core.inventory.inventory.Inventory.to_pyrocko_stations =\
        to_pyrocko_stations

    import pyrocko.trace
    import pyrocko.pile
    pyrocko.trace.Trace.to_obspy_trace = to_obspy_trace
    pyrocko.pile.Pile.to_obspy_stream = to_obspy_stream
