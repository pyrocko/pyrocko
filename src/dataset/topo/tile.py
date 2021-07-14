# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division

import math
import numpy as num
import scipy.signal

from pyrocko.orthodrome import positive_region


class OutOfBounds(Exception):
    pass


class Tile(object):

    def __init__(self, xmin, ymin, dx, dy, data):
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.dx = float(dx)
        self.dy = float(dy)
        self.data = data
        self._set_maxes()

    def _set_maxes(self):
        self.ny, self.nx = self.data.shape
        self.xmax = self.xmin + (self.nx-1) * self.dx
        self.ymax = self.ymin + (self.ny-1) * self.dy

    def x(self):
        return self.xmin + num.arange(self.nx) * self.dx

    def y(self):
        return self.ymin + num.arange(self.ny) * self.dy

    def decimate(self, ndeci):
        assert ndeci % 2 == 0
        kernel = num.ones((ndeci+1, ndeci+1))
        kernel /= num.sum(kernel)
        data = scipy.signal.convolve2d(
            self.data.astype(float), kernel, mode='valid')

        self.data = data[::ndeci, ::ndeci].astype(self.data.dtype)
        self.xmin += ndeci/2
        self.ymin += ndeci/2
        self.dx *= ndeci
        self.dy *= ndeci
        self._set_maxes()

    def yextend_with_repeat(self, ymin, ymax):
        assert ymax >= self.ymax
        assert ymin <= self.ymin

        nlo = int(round((self.ymin - ymin) / self.dy))
        nhi = int(round((ymax - self.ymax) / self.dy))

        nx, ny = self.nx, self.ny
        data = num.zeros((ny+nlo+nhi, nx), dtype=self.data.dtype)
        data[:nlo, :] = self.data[nlo, :]
        data[nlo:nlo+ny, :] = self.data
        data[nlo+ny:, :] = self.data[-1, :]

        self.ymin = ymin
        self.data = data
        self._set_maxes()

    def get(self, x, y):
        ix = int(round((x - self.xmin) / self.dx))
        iy = int(round((y - self.ymin) / self.dy))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return self.data[iy, ix]
        else:
            raise OutOfBounds()


def multiple_of(x, dx, eps=1e-5):
    return abs(int(round(x / dx))*dx - x) < dx * eps


def combine(tiles, region=None):
    if not tiles:
        return None

    dx = tiles[0].dx
    dy = tiles[0].dy
    dtype = tiles[0].data.dtype

    assert all(t.dx == dx for t in tiles)
    assert all(t.dy == dy for t in tiles)
    assert all(t.data.dtype == dtype for t in tiles)
    assert all(multiple_of(t.xmin, dx) for t in tiles)
    assert all(multiple_of(t.ymin, dy) for t in tiles)

    if region is None:
        xmin = min(t.xmin for t in tiles)
        xmax = max(t.xmax for t in tiles)
        ymin = min(t.ymin for t in tiles)
        ymax = max(t.ymax for t in tiles)
    else:
        xmin, xmax, ymin, ymax = positive_region(region)

        if not multiple_of(xmin, dx):
            xmin = math.floor(xmin / dx) * dx
        if not multiple_of(xmax, dx):
            xmax = math.ceil(xmax / dx) * dx
        if not multiple_of(ymin, dy):
            ymin = math.floor(ymin / dy) * dy
        if not multiple_of(ymax, dy):
            ymax = math.ceil(ymax / dy) * dy

    nx = int(round((xmax - xmin) / dx)) + 1
    ny = int(round((ymax - ymin) / dy)) + 1

    data = num.zeros((ny, nx), dtype=dtype)
    data[:, :] = 0

    for t in tiles:
        for txmin in (t.xmin, t.xmin + 360.):
            ix = int(round((txmin - xmin) / dx))
            iy = int(round((t.ymin - ymin) / dy))
            ixlo = max(ix, 0)
            ixhi = min(ix+t.nx, nx)
            iylo = max(iy, 0)
            iyhi = min(iy+t.ny, ny)
            jxlo = ixlo-ix
            jxhi = jxlo + max(0, ixhi - ixlo)
            jylo = iylo-iy
            jyhi = jylo + max(0, iyhi - iylo)
            if iyhi > iylo and ixhi > ixlo:
                data[iylo:iyhi, ixlo:ixhi] = t.data[jylo:jyhi, jxlo:jxhi]

    if not num.any(num.isfinite(data)):
        return None

    return Tile(xmin, ymin, dx, dy, data)
