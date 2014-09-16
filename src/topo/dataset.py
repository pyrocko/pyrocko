import math
import urllib2
import logging
import os
import os.path as op

import numpy as num

from pyrocko import util
from pyrocko.topo import tile

logger = logging.getLogger('pyrocko.topo.dataset')


class TiledGlobalDataset(object):

    def __init__(self, name, nx, ny, ntx, nty, dtype, data_dir=None,
                 citation=None, region=None):

        # dataset geometry (including overlap/endpoints)
        self.nx = int(nx)
        self.ny = int(ny)
        self.xmin = -180.
        self.xmax = 180.
        self.ymin = -90.
        self.ymax = 90.
        self.dx = (self.xmax-self.xmin) / (self.nx - 1)
        self.dy = (self.ymax-self.ymin) / (self.ny - 1)

        # tile geometry (including overlap)
        self.ntx = int(ntx)
        self.nty = int(nty)
        self.stx = (self.ntx - 1) * self.dx
        self.sty = (self.nty - 1) * self.dy

        self.ntilesx = (self.nx - 1) / (self.ntx - 1)
        self.ntilesy = (self.ny - 1) / (self.nty - 1)

        self.name = name
        self.dtype = dtype
        self.data_dir = data_dir
        self.citation = citation
        if region is not None:
            self.region = self.positive_region(region)
        else:
            self.region = None

    def covers(self, region):
        if self.region is None:
            return True

        a = self.region
        b = self.positive_region(region)
        return ((
            (a[0] <= b[0] and b[1] <= b[1]) or
            (a[0] <= b[0]+360. and b[1]+360 <= a[1]))
            and a[2] <= b[2] and b[3] <= a[3])

    def is_suitable(self, region, dmin, dmax):
        d = 360. / (self.nx - 1)
        return self.covers(region) and dmin <= d <= dmax

    def download_file(self, url, fpath):
        logger.info('starting download of %s' % url)

        util.ensuredirs(fpath)
        f = urllib2.urlopen(url)
        fpath_tmp = fpath + '.%i.temp' % os.getpid()
        g = open(fpath_tmp, 'wb')
        while True:
            data = f.read(1024)
            if not data:
                break
            g.write(data)

        g.close()
        f.close()

        os.rename(fpath_tmp, fpath)

        logger.info('finished download of %s' % url)

    def x(self):
        return self.xmin + num.arange(self.nx) * self.dx

    def y(self):
        return self.ymin + num.arange(self.ny) * self.dy

    def positive_region(self, region):
        xmin, xmax, ymin, ymax = [float(x) for x in region]

        assert -180. - 360. <= xmin < 180.
        assert -180. < xmax <= 180. + 360.
        assert -90. <= ymin < 90.
        assert -90. < ymax <= 90.

        if xmax < xmin:
            xmax += 360.

        if xmin < -180.:
            xmin += 360.
            xmax += 360.

        return (xmin, xmax, ymin, ymax)

    def tile_indices(self, region):
        xmin, xmax, ymin, ymax = self.positive_region(region)
        itxmin = int(math.floor((xmin - self.xmin) / self.stx))
        itxmax = int(math.ceil((xmax - self.xmin) / self.stx))
        itymin = int(math.floor((ymin - self.ymin) / self.sty))
        itymax = int(math.ceil((ymax - self.ymin) / self.sty))
        indices = []
        for ity in range(itymin, itymax):
            for itx in range(itxmin, itxmax):
                indices.append((itx % self.ntilesx, ity))

        return indices

    def get_tile(self, itx, ity):
        return None

    def get(self, region):
        indices = self.tile_indices(region)
        tiles = []
        for itx, ity in indices:
            t = self.get_tile(itx, ity)
            if t:
                tiles.append(t)

        return tile.combine(tiles, region)

    def get_with_repeat(self, region):
        xmin, xmax, ymin, ymax = region
        ymin2 = max(-90., ymin)
        ymax2 = min(90., ymax)
        region2 = xmin, xmax, ymin2, ymax2
        t = self.get(region2)
        if region2 != region:
            t.yextend_with_repeat(ymin, ymax)

        return t


class DecimatedTiledGlobalDataset(TiledGlobalDataset):

    def __init__(self, name, base, ndeci, data_dir=None, ntx=None, nty=None):

        assert ndeci % 2 == 0
        assert (base.nx - 1) % ndeci == 0
        assert (base.ny - 1) % ndeci == 0

        nx = (base.nx - 1) / ndeci + 1
        ny = (base.ny - 1) / ndeci + 1

        if ntx is None:
            ntx = base.ntx

        if nty is None:
            nty = base.nty

        assert (nx - 1) % (ntx - 1) == 0
        assert (ny - 1) % (nty - 1) == 0

        if data_dir is None:
            data_dir = op.join(base.data_dir, 'decimated_%i' % ndeci)

        TiledGlobalDataset.__init__(self, name, nx, ny, ntx, nty, base.dtype,
                                    data_dir=data_dir, citation=base.citation,
                                    region=base.region)

        self.base = base
        self.ndeci = ndeci

    def make_tile(self, itx, ity, fpath):
        nh = self.ndeci / 2
        xmin = self.xmin + itx*self.stx - self.base.dx * nh
        xmax = self.xmin + (itx+1)*self.stx + self.base.dx * nh
        ymin = self.ymin + ity*self.sty - self.base.dy * nh
        ymax = self.ymin + (ity+1)*self.sty + self.base.dy * nh

        t = self.base.get_with_repeat((xmin, xmax, ymin, ymax))
        if t is not None:
            t.decimate(self.ndeci)

        util.ensuredirs(fpath)
        with open(fpath, 'w') as f:
            if t is not None:
                t.data.tofile(f)

    def get_tile(self, itx, ity):
        assert 0 <= itx < self.ntilesx
        assert 0 <= ity < self.ntilesy

        fn = '%02i.%02i.bin' % (ity, itx)
        fpath = op.join(self.data_dir, fn)
        if not op.exists(fpath):
            logger.info('making decimated tile: %s (%s)' % (fn, self.name))
            self.make_tile(itx, ity, fpath)

        with open(fpath, 'r') as f:
            data = num.fromfile(f, dtype=self.dtype)

        if data.size == 0:
            return None

        assert data.size == self.ntx*self.nty
        data = data.reshape((self.ntx, self.nty))

        return tile.Tile(
            self.xmin + itx*self.stx,
            self.ymin + ity*self.sty,
            self.dx, self.dx, data)
