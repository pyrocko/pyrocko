from pyrocko import trace, util, model

import logging
import copy
import numpy as num
import cPickle as pickle

logger = logging.getLogger('pyrocko.eventdata')


class NoRestitution(Exception):
    pass


class FileNotFound(Exception):

    def __init__(self, s):
        self.s = s

    def __str__(self):
        return 'File not found: %s' % self.s


class Problems:
    def __init__(self):
        self._problems = {}

    def add(self, kind, nslct):
        if kind not in self._problems:
            self._problems[kind] = set()
        problem = self._problems[kind]
        problem.add(nslct)

    def dump(self, fn):
        f = open(fn, 'w')
        pickle.dump(self._problems, f)
        f.close()

    def load(self, fn):
        f = open(fn, 'r')
        self._problems = pickle.load(f)
        f.close()

    def mapped(self, mapping=lambda nslct: nslct[:3]):
        p = {}
        for kind, problem in self._problems.iteritems():
            nsl = set()
            for nslct in problem:
                nsl.add(mapping(nslct))
            p[kind] = nsl

        return p


class EventDataAccess:
    '''Abstract base class for event data access (see rdseed.py)'''

    def __init__(self, events=None, stations=None, datapile=None):

        self._pile = datapile
        self._events = events

        if stations is None:
            self._stations = None
        else:
            self._stations = {}
            for station in stations:
                self._stations[station.nsl()] = station

        self._problems = Problems()

    def get_pile(self):
        return self._pile

    def get_events(self):
        if not self._events:
            self._events = self._get_events_from_file()
        return self._events

    def get_event(self, i):
        return self.get_events()[i]

    def get_station(self, tr, relative_event=None):
        self._update_stations()
        s = copy.deepcopy(self._stations[tr.nslc_id[:3]])
        if relative_event is not None:
            s.set_event_relative_data(relative_event)
        return s

    def get_channel(self, tr):
        sta = self.get_station(tr)
        return sta.get_channel(tr.channel)

    def get_stations(self, relative_event=None):

        self._update_stations()

        stations = copy.deepcopy(self._stations)

        if relative_event is not None:
            for s in stations.values():
                s.set_event_relative_data(relative_event)

        return stations

    def _update_stations(self):
        if not self._stations:
            self._stations = {}
            for station in self._get_stations_from_file():
                self._stations[station.nsl()] = station
            self._insert_channel_descriptions(self._stations)

    def _insert_channel_descriptions(self, stations):
        pile = self.get_pile()
        nslc_ids = pile.gather_keys(
            lambda tr: (tr.network, tr.station, tr.location, tr.channel))

        for nslc in nslc_ids:
            if nslc[:3] not in stations:
                logger.warn(
                    'No station description for trace %s.%s.%s.%s' % nslc)
                continue

            sta = stations[nslc[:3]]
            try:
                sta.add_channel(self._get_channel_description_from_file(nslc))
            except FileNotFound:
                logger.warn(
                    'No channel description for trace %s.%s.%s.%s' % nslc)

    def _get_channel_description_from_file(self, nslc):
        return model.Channel(nslc[3], None, None, 1.)

    def iter_traces(self, group_selector=None, trace_selector=None):

        for traces in self.get_pile().chopper_grouped(
                gather=lambda tr: (tr.network, tr.station, tr.location),
                group_selector=group_selector,
                trace_selector=trace_selector):

            yield traces

    def problems(self):
        return self._problems

    def _redundant_channel_weeder(self, redundant_channel_priorities, nslcs):

        if redundant_channel_priorities is None:
            return []

        # n=network,s=station,l=location,c=channel
        # channels by station
        by_nsl = {}
        for nslc in nslcs:
            nsl = nslc[:3]
            if nsl not in by_nsl:
                by_nsl[nsl] = []

            by_nsl[nsl].append(nslc)

        # figure out what items to remove
        to_delete = []
        for ((h1, h2), (l1, l2)) in redundant_channel_priorities:
            for nsl, nslcs in by_nsl.iteritems():
                channels = [nslc[3] for nslc in nslcs]
                if h1 in channels and \
                        h2 in channels and \
                        l1 in channels and \
                        l2 in channels:

                    to_delete.append(nslc[:3] + (l1,))
                    to_delete.append(nslc[:3] + (l2,))

        return to_delete

    def iter_displacement_traces(
            self, tfade, freqband,
            deltat=None,
            rotations=None,
            projections=None,
            relative_event=None,
            maxdisplacement=None,
            extend=None,
            group_selector=None,
            trace_selector=None,
            allowed_methods=None,
            crop=True,
            out_stations=None,
            redundant_channel_priorities=None,
            restitution_off_hack=False,
            preprocess=None,
            progress='Processing traces'):

        stations = self.get_stations(relative_event=relative_event)
        if out_stations is not None:
            out_stations.clear()
        else:
            out_stations = {}

        for xtraces in self.get_pile().chopper_grouped(
                gather=lambda tr: (tr.network, tr.station, tr.location),
                group_selector=group_selector,
                trace_selector=trace_selector,
                progress=progress):

            xxtraces = []
            nslcs = set()
            for tr in xtraces:
                nsl = tr.network, tr.station, tr.location
                if nsl not in stations:
                    logger.warn(
                        'No station description for trace %s.%s.%s.%s' %
                        tr.nslc_id)
                    continue

                nslcs.add(tr.nslc_id)
                xxtraces.append(tr)

            to_delete = self._redundant_channel_weeder(
                redundant_channel_priorities, nslcs)

            traces = []
            for tr in xxtraces:
                if tr.nslc_id in to_delete:
                    logger.info(
                        'Skipping channel %s.%s.%s.%s due to redunancies.' %
                        tr.nslc_id)
                    continue
                traces.append(tr)

            traces.sort(lambda a, b: cmp(a.full_id, b.full_id))

            # mainly to get rid if overlaps and duplicates
            traces = trace.degapper(traces)
            if traces:
                nsl = traces[0].nslc_id[:3]
                # all traces belong to the same station here
                station = stations[nsl]

                displacements = []
                for tr in traces:

                    if preprocess is not None:
                        preprocess(tr)

                    tr.ydata = tr.ydata - num.mean(tr.ydata)

                    if deltat is not None:
                        try:
                            tr.downsample_to(
                                deltat, snap=True, allow_upsample_max=5)

                        except util.UnavailableDecimation, e:
                            self.problems().add(
                                'cannot_downsample', tr.full_id)
                            logger.warn(
                                'Cannot downsample %s.%s.%s.%s: %s' %
                                (tr.nslc_id + (e,)))
                            continue

                    try:
                        trans = self.get_restitution(tr, allowed_methods)
                    except NoRestitution, e:
                        self.problems().add('no_response', tr.full_id)
                        logger.warn(
                            'Cannot restitute trace %s.%s.%s.%s: %s' %
                            (tr.nslc_id + (e,)))

                        continue

                    try:
                        if extend:
                            tr.extend(
                                tr.tmin+extend[0],
                                tr.tmax+extend[1],
                                fillmethod='repeat')

                        if restitution_off_hack:
                            displacement = tr.copy()

                        else:
                            try:
                                displacement = tr.transfer(
                                    tfade,
                                    freqband,
                                    transfer_function=trans,
                                    cut_off_fading=crop)

                            except Exception, e:
                                if isinstance(e, trace.TraceTooShort):
                                    raise

                                logger.warn(
                                    'An error while applying transfer '
                                    'function to trace %s.%s.%s.%s.' %
                                    tr.nslc_id)

                                continue

                        amax = num.max(num.abs(displacement.get_ydata()))
                        if maxdisplacement is not None and \
                                amax > maxdisplacement:

                            self.problems().add(
                                'unrealistic_amplitude', tr.full_id)
                            logger.warn(
                                'Trace %s.%s.%s.%s has too large '
                                'displacement: %g' % (tr.nslc_id + (amax,)))

                            continue

                        if not num.all(num.isfinite(displacement.get_ydata())):
                            self.problems().add('has_nan_or_inf', tr.full_id)
                            logger.warn(
                                'Trace %s.%s.%s.%s has NaNs or Infs' %
                                tr.nslc_id)
                            continue

                    except trace.TraceTooShort, e:
                        self.problems().add('gappy', tr.full_id)
                        logger.warn('%s' % e)
                        continue

                    displacements.append(displacement)
                    if nsl not in out_stations:
                        out_stations[nsl] = copy.deepcopy(station)
                        out_station = out_stations[nsl]

                if displacements:
                    if projections:
                        for project in projections:
                            matrix, in_channels, out_channels = project(
                                out_station)
                            projected = trace.project(
                                displacements, matrix,
                                in_channels, out_channels)
                            displacements.extend(projected)
                            for tr in projected:
                                for ch in out_channels:
                                    if ch.name == tr.channel:
                                        out_station.add_channel(ch)

                    if rotations:
                        for rotate in rotations:
                            angle, in_channels, out_channels = \
                                rotate(out_station)
                            rotated = trace.rotate(
                                displacements, angle,
                                in_channels, out_channels)

                            displacements.extend(rotated)
                            for tr in rotated:
                                for ch in out_channels:
                                    if ch.name == tr.channel:
                                        out_station.add_channel(ch)

                yield displacements

    def get_restitution(self, tr, allowed_methods):
        if 'integration' in allowed_methods:
            trace.IntegrationResponse()
        else:
            raise Exception('only "integration" restitution method is allowed')
