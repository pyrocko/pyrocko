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
    '''
    Convert ObsPy inventory to list of Pyrocko traces
    '''
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
    '''
    Convert Pyrocko pile object to ObsPy stream
    '''
    import obspy
    stream = obspy.Stream()
    stream.extend([to_obspy_trace(tr) for tr in pyrocko_pile.iter_all()])
    return stream


def to_obspy_trace(pyrocko_trace):
    '''
    Convert Pyrocko trace to ObsPy trace
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


def load_stream_snuffling(win, obspy_stream):
    from pyrocko.gui import snuffling as sn

    class ObsPyStreamSnuffling(sn.Snuffling):
        '''Snuffling to fiddle with an ObsPy stream.'''

        def __init__(self, obspy_stream=None, *args, **kwargs):
            sn.Snuffling.__init__(self, *args, **kwargs)
            self.obspy_stream_orig = obspy_stream

        def setup(self):
            self.set_name('ObsPy Stream Fiddler')

            if len(obspy_stream) != 0:
                fmax = 0.5/min(tr.stats.delta for tr in obspy_stream)
                fmin = fmax / 1000.
            else:
                fmin = 0.001
                fmax = 1000.

            self.add_parameter(
                sn.Param(
                    'Highpass', 'highpass_corner', None, fmin, fmax,
                    low_is_none=True))
            self.add_parameter(
                sn.Param(
                    'Lowpass', 'lowpass_corner', None, fmin, fmax,
                    high_is_none=True))

        def init_gui(self, *args, **kwargs):
            sn.Snuffling.init_gui(self, *args, **kwargs)
            pyrocko_traces = to_pyrocko_traces(self.obspy_stream_orig)
            self.add_traces(pyrocko_traces)

        def call(self):
            try:
                obspy_stream = self.obspy_stream_orig.copy()
                if None not in (self.highpass_corner, self.lowpass_corner):
                    obspy_stream.filter(
                        'bandpass',
                        freqmin=self.highpass_corner,
                        freqmax=self.lowpass_corner)

                elif self.lowpass_corner is not None:
                    obspy_stream.filter(
                        'lowpass',
                        freq=self.lowpass_corner)

                elif self.highpass_corner is not None:
                    obspy_stream.filter(
                        'highpass',
                        freq=self.highpass_corner)

                self.cleanup()
                pyrocko_traces = to_pyrocko_traces(obspy_stream)
                self.add_traces(pyrocko_traces)
                self.obspy_stream = obspy_stream

            except Exception:
                raise  # logged by caller

    snuffling = ObsPyStreamSnuffling(obspy_stream=obspy_stream)
    # snuffling.setup()
    win.pile_viewer.viewer.add_snuffling(snuffling, reloaded=True)
    return snuffling


def snuffle(obspy_stream, obspy_inventory=None, obspy_catalog=None):
    '''
    Explore ObsPy data with Snuffler
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


def fiddle(obspy_stream, obspy_inventory=None, obspy_catalog=None):
    '''
    Manipulate stream object interactively.

    :param obspy_stream: :py:class:`obspy.Stream` object
    :param obspy_inventory: :py:class:`obspy.Inventory` object
    :param obspy_catalog: :py:class:`obspy.Catalog` object
    :returns: :py:class:`obspy.Stream` object with changes applied
        interactively

    This function displays an ObsPy stream object in Snuffler and adds a
    Snuffling panel to apply some basic ObsPy signal processing to the
    contained traces. The applied changes are handed back to the caller as
    a modified copy of the stream object.
    '''
    from pyrocko import trace
    import obspy

    events = to_pyrocko_events(obspy_catalog)
    stations = to_pyrocko_stations(obspy_inventory)

    if isinstance(obspy_stream, obspy.Trace):
        obspy_stream = obspy.Stream(traces=obspy_stream)

    snuffling = []

    def load_snuffling(win):
        snuffling[:] = [load_stream_snuffling(win, obspy_stream)]

    trace.snuffle(
        [],
        events=events,
        stations=stations,
        want_markers=True,
        launch_hook=load_snuffling)

    new_obspy_stream = snuffling[0].get_obspy_stream()
    if isinstance(obspy_stream, obspy.Trace):
        return new_obspy_stream[0]
    else:
        return new_obspy_stream


def plant():
    '''
    Add conversion functions as methods to ObsPy and Pyrocko classes.
    '''

    import obspy
    obspy.Trace.to_pyrocko_trace = to_pyrocko_trace
    obspy.Trace.snuffle = snuffle
    obspy.Stream.fiddle = fiddle

    obspy.Stream.to_pyrocko_traces = to_pyrocko_traces
    obspy.Stream.snuffle = snuffle
    obspy.Stream.fiddle = fiddle

    obspy.core.event.Catalog.to_pyrocko_events = to_pyrocko_events
    obspy.core.inventory.inventory.Inventory.to_pyrocko_stations =\
        to_pyrocko_stations

    import pyrocko.trace
    import pyrocko.pile
    pyrocko.trace.Trace.to_obspy_trace = to_obspy_trace
    pyrocko.pile.Pile.to_obspy_stream = to_obspy_stream
