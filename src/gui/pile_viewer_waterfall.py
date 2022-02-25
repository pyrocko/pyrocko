import logging
import hashlib
import numpy as num

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from pyrocko.trace import t2ind
from pyrocko.util import tts
from .qt_compat import qc, qg, qw

logger = logging.getLogger(__name__)


DEFAULT_CMAP = 'plasma'


class TraceWaterfall:

    def __init__(self):
        self.tmin = 0.
        self.tmax = 0.
        self.traces = []

        self.cmap = get_cmap(DEFAULT_CMAP)
        self.norm = Normalize()

        self._data_cache = None

        self._show_absolute = False
        self._clip_min = 0.
        self._clip_max = 1.
        self._common_scale = True

    def set_traces(self, traces):
        self.traces = traces

    def set_time_range(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax

    def set_clip(self, clip_min, clip_max):
        assert 0. <= clip_min < clip_max <= 1.
        self._clip_min = clip_min
        self._clip_max = clip_max

    def show_absolute_values(self, show_absolute):
        self._show_absolute = show_absolute

    @property
    def ntraces(self):
        return len(self.traces)

    def set_cmap(self, cmap):
        logger.debug('setting colormap to %s', cmap)
        self.cmap = get_cmap(cmap)

    def set_common_scale(self, _common_scale):
        self._common_scale = _common_scale

    def get_state_hash(self):
        sha1 = hashlib.sha1()
        sha1.update(self.tmin.hex().encode())
        sha1.update(self.tmax.hex().encode())
        sha1.update(self.cmap.name.encode())
        sha1.update(self._clip_min.hex().encode())
        sha1.update(self._clip_max.hex().encode())
        sha1.update(bytes(self._show_absolute))
        for tr in self.traces:
            sha1.update(bytes(tr.ydata[:10]))

        return sha1.hexdigest()

    def get_data(self):
        data_hash = self.get_state_hash()

        if self._data_cache and self._data_cache[-1] == data_hash:
            return self._data_cache

        dtypes = set(tr.ydata.dtype for tr in self.traces)
        dtype = num.float64 if num.float64 in dtypes else num.float32

        img_deltat = min(tr.deltat for tr in self.traces)
        img_nsamples = int(round((self.tmax - self.tmin) / img_deltat)) + 1

        data = num.zeros((self.ntraces, img_nsamples), dtype=dtype)
        empty_data = num.ones((self.ntraces, img_nsamples), dtype=num.bool)

        for itr, tr in enumerate(self.traces):
            tr_data = tr.ydata

            if tr.deltat != img_deltat:
                data_time_vec = tr.tmin + num.arange(tr.ydata.size)*img_deltat
                tr_data = num.interp(data_time_vec, tr.get_xdata(), tr.ydata)

            ibeg = max(0, t2ind(self.tmin - tr.tmin, img_deltat, round))
            iend = min(
                tr_data.size,
                t2ind(self.tmax - tr.tmin, img_deltat, round))
            tr_tmin = tr.tmin + ibeg * img_deltat

            img_ibeg = max(0, t2ind(tr_tmin - self.tmin, img_deltat, round))
            img_iend = img_ibeg + (iend - ibeg)

            data[itr, img_ibeg:img_iend] = tr_data[ibeg:iend]
            empty_data[itr, img_ibeg:img_iend] = False

        if self._common_scale:
            data /= num.abs(data).max(axis=1)[:, num.newaxis]

        vmax = num.abs(data).max()
        vmin = -vmax
        if self._show_absolute:
            data = num.abs(data)
            vmin = data.min()
        vrange = vmax - vmin

        self.norm.vmin = vmin + self._clip_min*vrange
        self.norm.vmax = vmax - (1. - self._clip_max)*vrange

        img_data = self.cmap(self.norm(data), alpha=None, bytes=True)

        # Mask out empty data
        img_data[empty_data, 3] = 0

        nx, ny = data.shape
        img = qg.QImage(
            img_data,
            ny, nx, qg.QImage.Format_RGBA8888)

        self._data_cache = (data, img, data_hash)
        return self._data_cache

    def draw_waterfall(self, p, rect=None):
        if not self.traces:
            raise AttributeError('No traces to paint.')

        trace_data, img, *_ = self.get_data()
        p.drawImage(rect or p.window(), img)
