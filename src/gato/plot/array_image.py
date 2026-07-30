from collections import defaultdict
import numpy as num
from matplotlib import pyplot as plt
from pyrocko import plot


def _configure(csmi_op, carpets):

    d_setups = dict(
        ((setup.array.name, setup.mapping.group_key), setup)
        for setup in csmi_op.get_setups())

    d_carpets = dict((carpet.codes, carpet) for carpet in carpets)

    d_out_codes = csmi_op.get_out_codes_by_agf()

    pages = defaultdict(list)
    for (array_name, group_key, field), codes in d_out_codes.items():
        if codes in d_carpets:
            pages[array_name, group_key].append(field)

    pages_data = []
    for (array_name, group_key), fields_avail in pages.items():

        setup = d_setups[array_name, group_key]

        t_projections = {}
        s_projections = {}
        for field in fields_avail:
            if field == 'avail':
                continue

            projection = field[:-4] if field.endswith('_max') else field

            source_grid = setup.generic_delay_table.source_grid
            coords = source_grid.native_coordinates_slices()[projection]
            dims = [arr.size for arr in coords]

            carpet = d_carpets[d_out_codes[array_name, group_key, field]]

            if len(dims) == 1 and dims[0] > 1:
                t_projections[field] = (carpet, source_grid)

            if len(dims) == 2 and dims[0] > 1 and dims[1] > 1:
                s_projections[field] = (carpet, source_grid)

        pages_data.append(
            (array_name, group_key, t_projections, s_projections))

    return pages_data


def unit_scale(unit_scales, unit):
    return unit_scales.get(unit, (unit, 1.0))


def get_projection(field):
    return field[:-4] if field.endswith('_max') else field


class ArrayImagePlot:
    def __init__(self, entries, unit_scales):
        self.entries = entries
        self.unit_scales = unit_scales
        self.artists = []
        self.fig = None
        self.setup()
        self.register_handlers()
        self.draw()
        self.connected = []

    def connect(self, event_name, handler):
        self.connected.append((event_name, handler))

    def emit(self, event_name, *args):
        for (event_name_, handler) in self.connected:
            if event_name_ == event_name:
                handler(*args)

    def on_mouse_down(self, event):
        pass
        # print(event)

    def on_mouse_up(self, event):
        pass

    def on_mouse_move(self, event):
        pass

    def register_handlers(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_down)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_up)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def clear_frame(self):
        while self.artists:
            self.artists.pop().remove()


class ArrayImageTimePlot(ArrayImagePlot):

    def __init__(self, *args, **kwargs):
        ArrayImagePlot.__init__(self, *args, **kwargs)
        self.selection_tmin = None

    def on_mouse_down(self, event):
        if event.xdata is not None:
            self.selection_tmin = event.xdata
            self.emit('set_time', event.xdata)

    def on_mouse_up(self, event):
        if event.xdata is not None and self.selection_tmin is not None:
            self.emit('set_time_range', self.selection_tmin, event.xdata)

        if event.xdata is None:
            self.emit('set_time', None)

        self.selection_tmin = None

    def on_mouse_move(self, event):
        if event.button == 1 and event.xdata is not None:
            if self.selection_tmin is not None:
                self.emit('set_time_range', self.selection_tmin, event.xdata)
            else:
                self.emit('set_time', event.xdata)

    def setup(self):
        fig = plt.figure(figsize=plot.mpl_papersize('a4', 'landscape'))
        d_axes = {}
        scales = {}
        axes = None
        for iaxes, (field, (_, source_grid)) in enumerate(
                self.entries.items()):

            d_axes[field] = axes = fig.add_subplot(
                len(self.entries), 1, iaxes+1, sharex=axes)

            plot.mpl_time_axis(axes)

            projection = get_projection(field)
            ylabel = source_grid.native_coordinate_labels()[projection]
            unit_orig = source_grid.native_coordinate_units()[projection]
            unit, scale = unit_scale(self.unit_scales, unit_orig)
            scales[projection] = scale
            axes.set_ylabel('%s [%s]' % (ylabel, unit))

        self.scales = scales
        self.d_axes = d_axes
        self.fig = fig

    def draw(self, tmin=None, tmax=None):

        self.clear_frame()

        for field, (carpet, source_grid) in self.entries.items():
            axes = self.d_axes[field]

            projection = get_projection(field)

            if carpet.ncomponents > 1:
                self.artists.append(
                    axes.pcolormesh(
                        carpet.times,
                        source_grid.native_coordinates()[projection]
                        / self.scales[projection],
                        carpet.data))

            elif carpet.ncomponents == 1:
                self.artists.extend(
                    axes.plot(carpet.times, carpet.data[0, :], color='black'))

        if tmin is not None and tmax is not None:
            for axes in self.d_axes.values():
                self.artists.append(axes.axvline(tmin, color='white'))
                self.artists.append(axes.axvline(tmax, color='white'))

        self.fig.canvas.draw()


class ArrayImageSectionPlot(ArrayImagePlot):

    def setup(self):
        fig = plt.figure(figsize=plot.mpl_papersize('a4', 'landscape'))
        d_axes = {}
        scales = {}
        axes = None
        for iaxes, (field, (_, source_grid)) in enumerate(
                self.entries.items()):

            projection = get_projection(field)
            xunit_orig = source_grid.native_coordinate_units()[projection[0]]
            xunit, xscale = unit_scale(self.unit_scales, xunit_orig)

            yunit_orig = source_grid.native_coordinate_units()[projection[1]]
            yunit, yscale = unit_scale(self.unit_scales, yunit_orig)

            if xunit == yunit:
                aspect = {'aspect': 1.0}
            else:
                aspect = {}

            d_axes[field] = axes = fig.add_subplot(
                len(self.entries), 1, iaxes+1,
                sharex=axes,
                sharey=axes,
                **aspect)

            xlabel, ylabel = [
                source_grid.native_coordinate_labels()[c] for c in projection]

            scales[projection[0]] = xscale
            scales[projection[1]] = yscale

            axes.set_xlabel('%s [%s]' % (xlabel, xunit))
            axes.set_ylabel('%s [%s]' % (ylabel, yunit))

        self.d_axes = d_axes
        self.fig = fig
        self.scales = scales

    def draw(self, tmin=None, tmax=None):

        self.clear_frame()

        for field, (carpet, source_grid) in self.entries.items():
            axes = self.d_axes[field]
            projection = get_projection(field)

            coords = source_grid.native_coordinates_slices()[projection]
            dims = [arr.size for arr in coords]

            itmin = None if tmin is None else carpet.itime(tmin, 'clip')
            itmax = None if tmax is None else carpet.itime(tmax, 'clip') + 1

            aggregate = num.mean

            data = aggregate(carpet.data[:, itmin:itmax], axis=1)

            data = data.reshape(dims)

            x, y = source_grid.native_coordinates_slice_grid(projection).T
            x = x.reshape(dims)
            y = y.reshape(dims)
            xscale = self.scales[projection[0]]
            yscale = self.scales[projection[1]]
            self.artists.append(
                axes.pcolormesh(x/xscale, y/yscale, data))

        self.fig.canvas.draw()


def plot_array_image(csmi_op, carpets):
    from pyrocko.cake import m2d

    pages_data = _configure(csmi_op, carpets)

    unit_scales = {
        # 's/m': ('s/km', 0.001),
        's/m': ('s/deg', m2d),
        'm': ('km', 1000)}

    plots = []
    for (array_name, group_key, t_entries, s_entries) in pages_data:
        title = 'Array: %s, Group: %s' % (array_name, group_key)
        t_plot = ArrayImageTimePlot(t_entries, unit_scales)
        t_plot.fig.suptitle(title)
        s_plot = ArrayImageSectionPlot(s_entries, unit_scales)
        s_plot.fig.suptitle(title)

        plots.append(t_plot)
        plots.append(s_plot)

    def set_time(time):
        for p in plots:
            p.draw(tmin=time, tmax=time)

    def set_time_range(tmin, tmax):
        for p in plots:
            p.draw(tmin=tmin, tmax=tmax)

    for p in plots:
        p.connect('set_time', set_time)
        p.connect('set_time_range', set_time_range)

    plt.show()


__all__ = [
    'plot_array_image',
]
