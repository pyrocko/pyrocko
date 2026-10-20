# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

# import copy
import logging
import operator
import calendar
import numpy as num

from pyrocko.guts import \
    Object, StringChoice, String, List, Float

from pyrocko import table, model  # , automap
from pyrocko.client import fdsn, catalog
from pyrocko.gui.qt_compat import qw, qc
from pyrocko.plot import beachball
# from pyrocko.himesh import HiMesh

# from pyrocko.gui.vtk_util import TrimeshPipe

from .table import TableElement, TableState

logger = logging.getLogger('pyrocko.gui.sparrow.elements.catalog')

guts_prefix = 'sparrow'


attribute_names = [
    'time', 'lat', 'lon', 'northing', 'easting', 'depth', 'magnitude']

attribute_dtypes = [
    'f16', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

name_to_icol = dict(
    (name, icol) for (icol, name) in enumerate(attribute_names))

# event_dtype = num.dtype(list(zip(attribute_names, attribute_dtypes)))

t_time = num.float64


def binned_statistic(values, ibins, function):
    order = num.argsort(ibins)
    values_sorted = values[order]
    ibins_sorted = ibins[order]
    parts = num.concatenate((
        [0],
        num.where(num.diff(ibins_sorted) != 0)[0] + 1,
        [ibins.size]))

    results = []
    ibins_result = []
    for ilow, ihigh in zip(parts[:-1], parts[1:]):
        values_part = values_sorted[ilow:ihigh]
        results.append(function(values_part))
        ibins_result.append(ibins_sorted[ilow])

    return num.array(ibins_result, dtype=num.int64), num.array(results)


def load_text(
        filepath,
        column_names=('time', 'lat', 'lon', 'depth', 'magnitude'),
        time_format='seconds'):

    with open(filepath, 'r') as f:
        if column_names == 'from_header':
            line = f.readline()
            column_names = line.split()

        name_to_icol_in = dict(
            (name, icol) for (icol, name) in enumerate(column_names)
            if name in attribute_names)

    data_in = num.loadtxt(filepath, skiprows=1)

    nevents = data_in.shape[0]
    c5 = num.zeros((nevents, 5))
    c5[:, 0] = data_in[:, name_to_icol_in['lat']]
    c5[:, 1] = data_in[:, name_to_icol_in['lon']]
    c5[:, 2] = 0.0
    c5[:, 3] = 0.0
    c5[:, 4] = data_in[:, name_to_icol_in['depth']] * 1000.

    tab = table.Table()
    loc_rec = table.LocationRecipe()
    tab.add_recipe(loc_rec)

    tab.add_col(loc_rec.c5_header, c5)
    for k, unit in [
            ('time', 's'),
            ('magnitude', None)]:

        values = data_in[:, name_to_icol_in[k]]

        if k == 'time' and time_format == 'year':
            values = decimal_year_to_time(values)

        tab.add_col(table.Header(k, unit), values)

    return tab


def decimal_year_to_time(year):
    iyear_start = num.floor(year).astype(num.int64)
    iyear_end = iyear_start + 1

    iyear_min = num.min(iyear_start)
    iyear_max = num.max(iyear_end)

    iyear_to_time = num.zeros(iyear_max - iyear_min + 1, dtype=t_time)
    for iyear in range(iyear_min, iyear_max+1):
        iyear_to_time[iyear-iyear_min] = calendar.timegm(
            (iyear, 1, 1, 0, 0, 0))

    tyear_start = iyear_to_time[iyear_start - iyear_min]
    tyear_end = iyear_to_time[iyear_end - iyear_min]

    t = tyear_start + (year - iyear_start) * (tyear_end - tyear_start)

    return t


def oa_to_array(objects, attribute):
    return num.fromiter(
        map(operator.attrgetter(attribute), objects),
        num.float64,
        len(objects))


def eventtags_to_array(events, tab):

    ks = set()
    event_tags = []
    for ev in events:
        tags = ev.tags_as_dict()
        ks.update(tags.keys())
        event_tags.append(tags)

    for k in sorted(ks):
        column = [tags.get(k, None) for tags in event_tags]
        if all(isinstance(v, int) for v in column):
            dtype = int
        elif all(isinstance(v, float) for v in column):
            dtype = float
        else:
            dtype = str
            column = [v or '' for v in column]

        arr = num.array(column, dtype=dtype)

        tab.add_col(table.Header('tag_%s' % k), arr)


def eventextras_to_array(events, tab):
    n_extras = num.array([len(ev.extras) for ev in events])
    evts_all_extras = num.arange(len(events))[n_extras != 0]

    if evts_all_extras.shape[0] == 0:
        return tab

    ev0 = events[evts_all_extras[0]]

    if num.unique(n_extras).shape[0] > 1:
        msg = 'Not all events have equal number of extras.'
        if num.unique(n_extras).shape[0] == 2 and num.min(n_extras) == 0:
            logger.warning(msg + ' Zero length lists are filled with NaNs.')
        else:
            raise IndexError(
                msg + ' Several non-zero shapes detected. Please check.')

    for key, val in ev0.extras.items():
        dtype = str
        values = num.array(['' for x in range(n_extras.shape[0])], dtype=dtype)

        if type(val) is float:
            dtype, values = num.float64, num.ones_like(n_extras) * num.nan

        elif type(val) is int:
            dtype = num.int64
            values = num.ones_like(n_extras, dtype=num.int64) * num.nan

        header = 'extra_%s' % key
        values[evts_all_extras] = num.array([
            events[iev].extras[key] for iev in evts_all_extras], dtype=dtype)

        tab.add_col(table.Header(header), values)


def events_to_table(events):
    c5 = num.zeros((len(events), 5))
    m6 = None

    for i, ev in enumerate(events):
        c5[i, :] = (
            ev.lat, ev.lon, ev.north_shift, ev.east_shift, ev.depth)

        if ev.moment_tensor:
            if m6 is None:
                m6 = num.zeros((len(events), 6))
                m6[:, 0:3] = 1.0

            m6[i, :] = beachball.deco_part(ev.moment_tensor, 'deviatoric').m6()

    tab = table.Table()

    ev_rec = table.EventRecipe()
    tab.add_recipe(ev_rec)
    tab.add_col(ev_rec.get_header('c5'), c5)

    for k in ['time', 'magnitude']:
        tab.add_col(ev_rec.get_header(k), oa_to_array(events, k))

    if events:
        eventtags_to_array(events, tab)
        eventextras_to_array(events, tab)

    if m6 is not None:
        mt_rec = table.MomentTensorRecipe()
        tab.add_recipe(mt_rec)
        tab.add_col(mt_rec.m6_header, m6)

    return tab


class LoadingChoice(StringChoice):
    choices = [choice.upper() for choice in [
        'file',
        'fdsn']]


class FDSNSiteChoice(StringChoice):
    choices = [key.upper() for key in fdsn.g_site_abbr.keys()]


class CatalogSelection(Object):

    def get_events(self):
        '''
        Get events of the selection.

        Returns ``None`` if the selection is not based on events (e.g. if it
        has been loaded from a plain text table).
        '''
        raise NotImplementedError()

    def get_table(self):
        return events_to_table(self.get_events())


class MemoryCatalogSelection(CatalogSelection):

    def __init__(self, events=None):
        if events is None:
            events = []

        self._events = events

    def get_events(self):
        return self._events


class FileCatalogSelection(CatalogSelection):
    paths = List.T(String.T())

    def _is_text_table(self):
        return any(
            path.split('.')[-1].lower() in ['dat', 'csv']
            for path in self.paths)

    def get_events(self):
        from pyrocko.io import quakeml

        if self._is_text_table():
            return None

        events = []
        for path in self.paths:
            fn_ext = path.split('.')[-1].lower()
            if fn_ext in ['xml', 'qml', 'quakeml']:
                qml = quakeml.QuakeML.load_xml(filename=path)
                events.extend(qml.get_pyrocko_events())
            else:
                events.extend(model.load_events(path))

        return events

    def get_table(self):
        if self._is_text_table():
            assert len(self.paths) == 1
            return load_text(
                self.paths[0], column_names='from_header', time_format='year')

        return events_to_table(self.get_events())


g_catalogs = {
    'Geofon': catalog.Geofon(),
    'USGS/NEIC US': catalog.USGS('us'),
    'Global-CMT': catalog.GlobalCMT(),
    'Saxony (Uni-Leipzig)': catalog.Saxony()
}

g_fdsn_has_events = ['ISC', 'SCEDC', 'NCEDC', 'IRIS', 'GEONET']

g_sites = sorted(g_catalogs.keys())
g_sites.extend(g_fdsn_has_events)


class OnlineCatalogSelection(CatalogSelection):
    site = String.T()
    tmin = Float.T()
    tmax = Float.T()
    magnitude_min = Float.T(optional=True)

    def get_events(self):
        logger.info('Getting events from "%s" catalog.' % self.site)

        cat = g_catalogs.get(self.site, None)
        try:
            if cat:
                kwargs = {}
                if self.magnitude_min is not None:
                    kwargs['magmin'] = self.magnitude_min

                events = cat.get_events(
                    time_range=(self.tmin, self.tmax), **kwargs)

            else:
                kwargs = {}
                if self.magnitude_min is not None:
                    kwargs['minmagnitude'] = self.magnitude_min

                request = fdsn.event(
                    starttime=self.tmin,
                    endtime=self.tmax,
                    site=self.site, **kwargs)

                from pyrocko.io import quakeml
                qml = quakeml.QuakeML.load_xml(request)
                events = qml.get_pyrocko_events()

            logger.info('Got %i event%s from "%s" catalog.' % (
                len(events), '' if len(events) == 1 else 's', self.site))

        except Exception as e:
            logger.error(
                'Getting events from "%s" catalog failed: %s' % (
                    self.site, str(e)))

            events = []

        return events


_m6_indices = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def m6s_to_m33s(m6s):
    m33s = num.zeros((m6s.shape[0], 3, 3))
    for i, (j, k) in enumerate(_m6_indices):
        m33s[:, j, k] = m6s[:, i]
        m33s[:, k, j] = m6s[:, i]

    return m33s


def m33s_to_m6s(m33s):
    return num.stack([
        m33s[:, j, k]
        for (j, k) in _m6_indices],
        axis=1)


def skew_matrices(axes):
    k = num.zeros((axes.shape[0], 3, 3))
    k[:, 0, 1] = -axes[:, 2]
    k[:, 0, 2] = axes[:, 1]
    k[:, 1, 0] = axes[:, 2]
    k[:, 1, 2] = -axes[:, 0]
    k[:, 2, 0] = -axes[:, 1]
    k[:, 2, 1] = axes[:, 0]
    return k


class MomentTensorInterpolator(object):
    '''
    Interpolate between pairs of moment tensors.

    The eigenvalues (normalized to unit norm) are interpolated linearly, the
    scalar moment logarithmically and the orientation of the principal axes
    along the shortest rotation connecting the two eigensystems (taking into
    account the sign ambiguity of the eigenvectors).
    '''

    def __init__(self, m6s_a, m6s_b):
        evals_a, rots_a = num.linalg.eigh(m6s_to_m33s(m6s_a))
        evals_b, rots_b = num.linalg.eigh(m6s_to_m33s(m6s_b))

        # make proper rotations
        rots_a[:, :, 2] *= num.sign(num.linalg.det(rots_a))[:, num.newaxis]
        rots_b[:, :, 2] *= num.sign(num.linalg.det(rots_b))[:, num.newaxis]

        # choose the eigenvector signs giving the smallest rotation
        best_trace = num.full(m6s_a.shape[0], -num.inf)
        best_rots = num.zeros_like(rots_a)
        for flips in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
            rots = num.einsum(
                'nij,nkj->nik',
                rots_b * num.array(flips)[num.newaxis, num.newaxis, :],
                rots_a)

            trace = num.einsum('nii->n', rots)
            better = trace > best_trace
            best_trace[better] = trace[better]
            best_rots[better] = rots[better]

        angles = num.arccos(num.clip((best_trace - 1.0) * 0.5, -1.0, 1.0))
        axes = num.stack([
            best_rots[:, 2, 1] - best_rots[:, 1, 2],
            best_rots[:, 0, 2] - best_rots[:, 2, 0],
            best_rots[:, 1, 0] - best_rots[:, 0, 1]], axis=1)

        axes_norm = num.sqrt(num.sum(axes**2, axis=1))
        axes_norm[axes_norm == 0.0] = 1.0
        axes /= axes_norm[:, num.newaxis]

        moments_a = num.sqrt(num.sum(evals_a**2, axis=1))
        moments_b = num.sqrt(num.sum(evals_b**2, axis=1))
        moments_a[moments_a == 0.0] = 1.0
        moments_b[moments_b == 0.0] = 1.0

        self._evals_a = evals_a / moments_a[:, num.newaxis]
        self._evals_b = evals_b / moments_b[:, num.newaxis]
        self._log_moments_a = num.log(moments_a)
        self._log_moments_b = num.log(moments_b)
        self._rots_a = rots_a
        self._angles = angles
        self._skew = skew_matrices(axes)

    def __call__(self, x):
        evals = (1.0 - x) * self._evals_a + x * self._evals_b
        evals_norm = num.sqrt(num.sum(evals**2, axis=1))
        evals_norm[evals_norm == 0.0] = 1.0
        evals *= (num.exp(
            (1.0 - x) * self._log_moments_a
            + x * self._log_moments_b) / evals_norm)[:, num.newaxis]

        angles = x * self._angles
        k = self._skew
        rots = num.eye(3)[num.newaxis, :, :] \
            + num.sin(angles)[:, num.newaxis, num.newaxis] * k \
            + (1.0 - num.cos(angles))[:, num.newaxis, num.newaxis] \
            * num.einsum('nij,njk->nik', k, k)

        rots = num.einsum('nij,njk->nik', rots, self._rots_a)

        m33s = num.einsum('nij,nj,nkj->nik', rots, evals, rots)
        return m33s_to_m6s(m33s)


def events_to_m6s(events):
    m6s = num.zeros((len(events), 6))
    has_mt = num.zeros(len(events), dtype=bool)
    for i, ev in enumerate(events):
        if ev.moment_tensor:
            m6s[i, :] = beachball.deco_part(
                ev.moment_tensor, 'deviatoric').m6()
            has_mt[i] = True

    return m6s, has_mt


def events_to_locations(events):
    lats = oa_to_array(events, 'effective_lat')
    lons = oa_to_array(events, 'effective_lon')
    depths = oa_to_array(events, 'depth')
    return lats, lons, depths


class CatalogInterpolator(object):
    '''
    Interpolate between two solutions for the same events.

    Events in the secondary catalog are associated with events in the primary
    catalog by event name. Events without counterpart are left unchanged.
    '''

    def __init__(self, events_a, events_b):
        name_to_event_b = {}
        for ev in events_b:
            if ev.name:
                if ev.name in name_to_event_b:
                    logger.warning(
                        'Duplicate event name in secondary catalog: %s'
                        % ev.name)
                else:
                    name_to_event_b[ev.name] = ev

        events_b_matched = []
        nmatched = 0
        for ev in events_a:
            ev_b = name_to_event_b.get(ev.name, None) if ev.name else None
            if ev_b is None:
                events_b_matched.append(ev)
            else:
                events_b_matched.append(ev_b)
                nmatched += 1

        logger.info(
            'Matched %i of %i event%s with secondary catalog.' % (
                nmatched, len(events_a), '' if len(events_a) == 1 else 's'))

        events_b = events_b_matched

        self._lats_a, self._lons_a, self._depths_a = \
            events_to_locations(events_a)
        lats_b, lons_b, depths_b = events_to_locations(events_b)
        self._dlats = lats_b - self._lats_a
        self._dlons = (lons_b - self._lons_a + 180.) % 360. - 180.
        self._ddepths = depths_b - self._depths_a

        self._times_a = oa_to_array(events_a, 'time')
        self._dtimes = oa_to_array(events_b, 'time') - self._times_a

        self._magnitudes_a = num.array(
            [ev.magnitude for ev in events_a], dtype=float)
        magnitudes_b = num.array(
            [ev.magnitude for ev in events_b], dtype=float)
        magnitudes_b[num.isnan(magnitudes_b)] = \
            self._magnitudes_a[num.isnan(magnitudes_b)]
        self._magnitudes_a[num.isnan(self._magnitudes_a)] = \
            magnitudes_b[num.isnan(self._magnitudes_a)]
        self._dmagnitudes = magnitudes_b - self._magnitudes_a

        m6s_a, has_mt_a = events_to_m6s(events_a)
        m6s_b, has_mt_b = events_to_m6s(events_b)
        if num.any(has_mt_a) or num.any(has_mt_b):
            m6s_a[~has_mt_a] = m6s_b[~has_mt_a]
            m6s_b[~has_mt_b] = m6s_a[~has_mt_b]
            has_mt = has_mt_a | has_mt_b
            m6s_a[~has_mt, 0:3] = 1.0
            m6s_b[~has_mt, 0:3] = 1.0
            self._m6s_a = m6s_a
            self._m6_interpolator = MomentTensorInterpolator(m6s_a, m6s_b)
        else:
            self._m6s_a = None
            self._m6_interpolator = None

        tab_a = events_to_table(events_a)
        self._extra_cols = [
            (name, tab_a.get_col(name))
            for name in tab_a.get_col_names(sub_headers=False)
            if name.startswith('tag_') or name.startswith('extra_')]

    def get_table(self, x):
        nevents = self._times_a.size
        c5 = num.zeros((nevents, 5))
        c5[:, 0] = self._lats_a + x * self._dlats
        c5[:, 1] = self._lons_a + x * self._dlons
        c5[:, 4] = self._depths_a + x * self._ddepths

        tab = table.Table()

        ev_rec = table.EventRecipe()
        tab.add_recipe(ev_rec)
        tab.add_col(ev_rec.get_header('c5'), c5)
        tab.add_col(
            ev_rec.get_header('time'), self._times_a + x * self._dtimes)
        tab.add_col(
            ev_rec.get_header('magnitude'),
            self._magnitudes_a + x * self._dmagnitudes)

        for name, values in self._extra_cols:
            tab.add_col(table.Header(name), values)

        if self._m6_interpolator is not None:
            mt_rec = table.MomentTensorRecipe()
            tab.add_recipe(mt_rec)
            if x == 0.0:
                m6s = self._m6s_a
            else:
                m6s = self._m6_interpolator(x)

            tab.add_col(mt_rec.m6_header, m6s)

        return tab


class CatalogState(TableState):
    selection = CatalogSelection.T(optional=True)
    secondary_selection = CatalogSelection.T(optional=True)
    interpolation = Float.T(default=0.0)
    size_parameter = String.T(default='magnitude', optional=True)
    color_parameter = String.T(default='depth', optional=True)

    @classmethod
    def get_name(self):
        return 'Catalog'

    def create(self):
        element = CatalogElement()
        return element


class CatalogElement(TableElement):

    def __init__(self, *args, **kwargs):
        TableElement.__init__(self, *args, **kwargs)
        self._selection_view = None
        self._secondary_selection_view = None
        self._interpolation_view = None
        self._events = None
        self._interpolator = None
        self._interpolation_slider = None
        # self._himesh = HiMesh(order=6)

        # cpt_data = [
        #     (0.0, 0.0, 0.0, 0.0),
        #     (1.0, 0.9, 0.9, 0.2)]
        #
        # self.cpt_mesh = automap.CPT(
        #     levels=[
        #         automap.CPTLevel(
        #             vmin=a[0],
        #             vmax=b[0],
        #             color_min=[255*x for x in a[1:]],
        #             color_max=[255*x for x in b[1:]])
        #         for (a, b) in zip(cpt_data[:-1], cpt_data[1:])])

    def get_name(self):
        return 'Catalog'

    def get_size_parameter_extra_entries(self):
        return ['magnitude']

    def get_color_parameter_extra_entries(self):
        return ['depth']

    def bind_state(self, state):
        TableElement.bind_state(self, state)
        self.talkie_connect(
            state,
            ['selection', 'secondary_selection', 'interpolation'],
            self.update)

    def update(self, *args):
        state = self._state
        # ifaces = None
        selection_changed = self._selection_view is not state.selection
        if selection_changed:
            if state.selection is not None:
                self._events = state.selection.get_events()
            else:
                self._events = None

            self._selection_view = state.selection

            # ifaces = self._himesh.points_to_faces(self._table.get_col('xyz'))

        if selection_changed \
                or self._secondary_selection_view \
                is not state.secondary_selection:

            self._interpolator = None
            if state.secondary_selection is not None:
                if self._events is None:
                    logger.warning(
                        'Interpolation between catalogs requires event '
                        'based primary catalog.')
                else:
                    events_b = state.secondary_selection.get_events()
                    if events_b is None:
                        logger.warning(
                            'Interpolation between catalogs requires event '
                            'based secondary catalog.')
                    else:
                        self._interpolator = CatalogInterpolator(
                            self._events, events_b)

            self._secondary_selection_view = state.secondary_selection

            if state.selection is not None:
                if self._interpolator is not None:
                    self.set_table(
                        self._interpolator.get_table(state.interpolation))
                elif self._events is not None:
                    self.set_table(events_to_table(self._events))
                else:
                    self.set_table(state.selection.get_table())

            self._interpolation_view = state.interpolation
            self._update_interpolation_controls()

        elif self._interpolator is not None \
                and self._interpolation_view != state.interpolation:

            self.set_table(
                self._interpolator.get_table(state.interpolation),
                update_controls=False)

            self._interpolation_view = state.interpolation

        TableElement.update(self, *args)

        # if ifaces is not None:
        #     ifaces_x, sizes = binned_statistic(
        #         ifaces, ifaces, lambda part: part.shape[0])
        #
        #     vertices = self._himesh.get_vertices()
        #     # vertices *= 0.95
        #     faces = self._himesh.get_faces()
        #
        #     values = num.zeros(faces.shape[0])
        #     values[ifaces_x] = num.log(1+sizes)
        #
        #     self._mesh = TrimeshPipe(vertices, faces, values=values)
        #     cpt = copy.deepcopy(self.cpt_mesh)
        #     cpt.scale(num.min(values), num.max(values))
        #     self._mesh.set_cpt(cpt)
        #     self._mesh.set_opacity(0.5)

        #     self._parent.add_actor(self._mesh.actor)

    def _update_interpolation_controls(self):
        if self._interpolation_slider is not None:
            self._interpolation_slider.setEnabled(
                self._interpolator is not None)

    def open_file_dialog(self, target='selection'):
        caption = 'Select one or more files to open'

        fns, _ = qw.QFileDialog.getOpenFileNames(
            self._parent, caption)

        if fns:
            setattr(self._state, target, FileCatalogSelection(
                paths=[str(fn) for fn in fns]))

    def clear_secondary_selection(self):
        self._state.secondary_selection = None

    def open_catalog_load_dialog(self, target='selection'):
        dialog = qw.QDialog(self._parent)
        dialog.setWindowTitle('Get events from online catalog.')

        layout = qw.QHBoxLayout(dialog)

        layout.addWidget(qw.QLabel('Site'))

        cb = qw.QComboBox()
        for i, s in enumerate(g_sites):
            cb.insertItem(i, s)

        layout.addWidget(cb)

        pb = qw.QPushButton('Cancel')
        pb.clicked.connect(dialog.reject)
        layout.addWidget(pb)

        pb = qw.QPushButton('Ok')
        pb.clicked.connect(dialog.accept)
        layout.addWidget(pb)

        dialog.exec_()

        site = str(cb.currentText())

        vstate = self._parent.state

        if dialog.result() == qw.QDialog.Accepted:
            setattr(self._state, target, OnlineCatalogSelection(
                site=site,
                tmin=vstate.tmin,
                tmax=vstate.tmax))

    def _get_table_widgets_start(self):
        return 3  # used as y arg in addWidget calls

    def _get_controls(self):
        if not self._controls:
            from ..state import state_bind_slider

            frame = TableElement._get_controls(self)  # sets self._controls
            layout = frame.layout()

            for iy, (label, target) in enumerate([
                    ('Load from', 'selection'),
                    ('Secondary', 'secondary_selection')]):

                layout.addWidget(qw.QLabel(label), iy, 0)

                pb = qw.QPushButton('File')
                layout.addWidget(pb, iy, 1)
                pb.clicked.connect(
                    lambda _, target=target: self.open_file_dialog(target))

                pb = qw.QPushButton('Online Catalog')
                layout.addWidget(pb, iy, 2)
                pb.clicked.connect(
                    lambda _, target=target: self.open_catalog_load_dialog(
                        target))

            layout.addWidget(qw.QLabel('Interpolation'), 2, 0)

            slider = qw.QSlider(qc.Qt.Horizontal)
            slider.setSizePolicy(
                qw.QSizePolicy(
                    qw.QSizePolicy.Expanding, qw.QSizePolicy.Fixed))
            slider.setMinimum(0)
            slider.setMaximum(1000)
            layout.addWidget(slider, 2, 1)
            state_bind_slider(
                self, self._state, 'interpolation', slider, factor=0.001)

            pb = qw.QPushButton('Clear Secondary')
            layout.addWidget(pb, 2, 2)
            pb.clicked.connect(self.clear_secondary_selection)

            self._interpolation_slider = slider
            self._update_interpolation_controls()

        return self._controls


__all__ = [
    'CatalogSelection',
    'FileCatalogSelection',
    'MemoryCatalogSelection',
    'CatalogElement',
    'CatalogState',
]
