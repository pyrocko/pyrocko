from __future__ import division, print_function, absolute_import
import os
import math
import tempfile
import shutil
import unittest
import numpy as num
from numpy.testing import assert_allclose
from matplotlib import image, pyplot as plt

from pyrocko import util
from pyrocko.plot import gmtpy
from pyrocko.plot.gmtpy import cm, inch, golden_ratio

from . import common

plot = False


@unittest.skipUnless(
    gmtpy.have_gmt(), 'GMT not available')
@unittest.skipUnless(
    gmtpy.have_pixmaptools(), '`pdftocairo` or `convert` not available')
class GmtPyTestCase(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def fpath(self, fn):
        return os.path.join(self.tempdir, fn)

    def fpath_ref(self, fn):
        try:
            return common.test_data_file(fn)
        except util.DownloadError:
            return common.test_data_file_no_download(fn)

    def compare_with_ref(self, fname, tolerance, show=False):
        fpath = self.fpath(fname)
        fpath_ref = self.fpath_ref(fname)

        if not os.path.exists(fpath_ref):
            shutil.copy(fpath, fpath_ref)

        img = image.imread(fpath)
        img_ref = image.imread(fpath_ref)
        self.assertEqual(img.shape, img_ref.shape)
        d = num.abs(img - img_ref)
        merr = num.mean(d)
        if (merr > tolerance or show) and plot:
            fig = plt.figure()
            axes1 = fig.add_subplot(1, 3, 1, aspect=1.)
            axes2 = fig.add_subplot(1, 3, 2, aspect=1.)
            axes3 = fig.add_subplot(1, 3, 3, aspect=1.)
            axes1.imshow(img)
            axes1.set_title('Candidate')
            axes2.imshow(img_ref)
            axes2.set_title('Reference')
            axes3.imshow(d)
            axes3.set_title('Mean abs difference: %g' % merr)
            plt.show()
            plt.close(fig)

        assert merr <= tolerance

    def test_basic(self):
        for version in gmtpy.all_installed_gmt_versions():
            width = 8.0 * inch
            height = 9.0 * inch
            resolution = 72

            gmt = gmtpy.GMT(version=version, config_papersize=(width, height))

            gmt.pscoast(
                X=0,
                Y=0,
                R='g',
                J='E32/30/170/8i',
                B='10g10',
                D='c',
                A=10000,
                S=(114, 159, 207),
                G=(233, 185, 110),
                W='thinnest')

            gmt.dump('test')
            gmt.load('test')

            for oversample in (1, 2):
                fname = 'gmtpy_test_basic_o%i.png' % oversample
                fpath = self.fpath(fname)
                gmt.save(fpath, resolution=resolution, oversample=oversample)

                self.compare_with_ref(fname, 0.03)

                img = image.imread(fpath, format='png')
                self.assertEqual(img.shape, (
                    int(round(resolution*height/inch)),
                    int(round(resolution*width/inch)), 3))

    def test_basic2(self):
        for version in gmtpy.all_installed_gmt_versions():
            if version.startswith('5'):
                gmt = gmtpy.GMT(
                    version=version,
                    config={'MAP_FRAME_TYPE': 'fancy'},
                    eps_mode=True)
            else:
                gmt = gmtpy.GMT(
                    version=version,
                    config={'BASEMAP_TYPE': 'fancy'})

            layout = gmt.default_layout()
            widget = layout.get_widget()

            xax = gmtpy.Ax(label='Lon', mode='min-max')
            yax = gmtpy.Ax(label='Lat', mode='min-max')
            scaler = gmtpy.ScaleGuru([([5, 15], [52, 58])], axes=(xax, yax))

            par = scaler.get_params()
            lon0 = (par['xmin'] + par['xmax'])/2.
            lat0 = (par['ymin'] + par['ymax'])/2.
            sll = '%g/%g' % (lon0, lat0)
            widget['J'] = '-JM' + sll + '/%(width)gp'

            widget['J'] = '-JM' + sll + '/%(width)gp'
            scaler['B'] = \
                '-B%(xinc)gg%(xinc)g:%(xlabel)s:' \
                '/%(yinc)gg%(yinc)g:%(ylabel)s:WSen'

            aspect = gmtpy.aspect_for_projection(
                version, *(widget.J() + scaler.R()))

            aspect = 1.045
            widget.set_aspect(aspect)

            gmt.pscoast(D='h', W='1p,red', *(widget.JXY() + scaler.R()))
            gmt.psbasemap(*(widget.JXY() + scaler.BR()))

            fname = 'gmtpy_test_basic2.png'
            fpath = self.fpath(fname)
            gmt.save(fpath, resolution=75, bbox=layout.bbox())

            self.compare_with_ref(fname, 0.01, show=False)

    def test_layout(self):
        x = num.linspace(0., math.pi*6, 1001)
        y1 = num.sin(x) * 1e-9
        y2 = 2.0 * num.cos(x) * 1e-9

        xax = gmtpy.Ax(label='Time', unit='s')
        yax = gmtpy.Ax(
            label='Amplitude', unit='m', scaled_unit='nm',
            scaled_unit_factor=1e9, approx_ticks=5, space=0.05)

        guru = gmtpy.ScaleGuru([(x, y1), (x, y2)], axes=(xax, yax))

        for version in gmtpy.all_installed_gmt_versions():
            width = 8*inch
            height = 3*inch

            gmt = gmtpy.GMT(
                version=version,
                config_papersize=(width, height))

            layout = gmt.default_layout()
            widget = layout.get_widget()

            gmt.draw_layout(layout)

            gmt.psbasemap(*(widget.JXY() + guru.RB(ax_projection=True)))
            gmt.psxy(
                in_columns=(x, y1), W='1p,red', *(widget.JXY() + guru.R()))
            gmt.psxy(
                in_columns=(x, y2), W='1p,blue', *(widget.JXY() + guru.R()))

            fname = 'gmtpy_test_layout.png'
            fpath = self.fpath(fname)
            gmt.save(fpath)

            self.compare_with_ref(fname, 0.01)

    def test_grid_layout(self):
        for version in gmtpy.all_installed_gmt_versions():
            gmt = gmtpy.GMT(version=version, config_papersize='a3')
            nx, ny = 2, 5
            grid = gmtpy.GridLayout(nx, ny)

            layout = gmt.default_layout()
            layout.set_widget('center', grid)

            widgets = []
            for iy in range(ny):
                for ix in range(nx):
                    inner = gmtpy.FrameLayout()
                    inner.set_fixed_margins(
                        1.*cm*golden_ratio, 1.*cm*golden_ratio, 1.*cm, 1.*cm)

                    grid.set_widget(ix, iy, inner)
                    inner.set_vertical(0, (iy+1.))
                    widgets.append(inner.get_widget('center'))

            gmt.draw_layout(layout)
            for widget in widgets:
                x = num.linspace(0., 10., 5)
                y = num.sin(x)
                xax = gmtpy.Ax(approx_ticks=4, snap=True)
                yax = gmtpy.Ax(approx_ticks=4, snap=True)
                guru = gmtpy.ScaleGuru([(x, y)], axes=(xax, yax))
                gmt.psbasemap(*(widget.JXY() + guru.RB(ax_projection=True)))
                gmt.psxy(in_columns=(x, y), *(widget.JXY() + guru.R()))

            fname = 'gmtpy_test_grid_layout.png'
            fpath = self.fpath(fname)
            gmt.save(fpath, resolution=75)

            self.compare_with_ref(fname, 0.01)

    def test_simple(self):

        x = num.linspace(0., 2*math.pi)
        y = num.sin(x)
        y2 = num.cos(x)

        for version in gmtpy.all_installed_gmt_versions():
            for ymode in ['off', 'symmetric', 'min-max', 'min-0', '0-max']:
                plot = gmtpy.Simple(gmtversion=version, ymode=ymode)
                plot.plot((x, y), '-W1p,%s' % gmtpy.color('skyblue2'))
                plot.plot((x, y2), '-W1p,%s' % gmtpy.color(
                    gmtpy.color_tup('scarletred2')))
                plot.text((3., 0.5, 'hello'), size=20.)
                fname = 'gmtpy_test_simple_%s.png' % ymode
                fpath = self.fpath(fname)
                plot.save(fpath)
                self.compare_with_ref(fname, 0.01, show=False)

    @unittest.skip('won\'t-fix-this')
    def test_simple_density(self):
        x = num.linspace(0., 2.*math.pi, 50)
        y = num.linspace(0., 2.*math.pi, 50)

        x2 = num.tile(x, y.size)
        y2 = num.repeat(y, x.size)
        z2 = num.sin(x2) * num.sin(y2)

        for version in gmtpy.all_installed_gmt_versions():
            for method in ['surface', 'triangulate', 'fillcontour']:
                plot = gmtpy.Simple(gmtversion=version, with_palette=True)
                plot.density_plot((x2, y2, z2), method=method)
                fname = 'gmtpy_test_simple_density_%s.png' % method
                fpath = self.fpath(fname)
                plot.save(fpath)
                self.compare_with_ref(fname, 0.02)

    def test_grid_data(self):
        x = num.linspace(0., 2.*math.pi, 100)
        y = num.linspace(0., 2.*math.pi, 100)

        x2 = num.tile(x, y.size)
        y2 = num.repeat(y, x.size)
        z2 = num.sin(x2) * num.sin(y2)

        xf, yf, zf = gmtpy.griddata_auto(x2, y2, z2)
        assert (xf.size, yf.size, zf.size) == (100, 100, 100*100)
        x3, y3, z3 = gmtpy.tabledata(xf, yf, zf)

        assert_allclose(x3, x2, atol=1e-7)
        assert_allclose(y3, y2, atol=1e-7)
        assert_allclose(z3, z2, atol=1e-7)

        xf2, yf2, zf2 = gmtpy.doublegrid(xf, yf, zf)
        assert (xf2.size, yf2.size, zf2.size) == (199, 199, 199*199)

        fn = self.fpath('grid.nc')
        for naming in ['xy', 'lonlat']:
            gmtpy.savegrd(xf, yf, zf, fn, naming=naming, title='mygrid')
            xf3, yf3, zf3 = gmtpy.loadgrd(fn)
            assert_allclose(xf3, xf)
            assert_allclose(yf3, yf)
            assert_allclose(zf3, zf)

    def test_text_box(self):

        for version in gmtpy.all_installed_gmt_versions():
            s = gmtpy.text_box('Hello', gmtversion=version)
            assert_allclose(s, (25.8, 9.), rtol=0.1)
            s = gmtpy.text_box(
                'Abc def ghi jkl mno pqr stu vwx yz',
                gmtversion=version)
            assert_allclose(s, (179.9, 12.3), rtol=0.01)

    def test_override_args(self):
        x = num.array([0, 0.5, 1, 0])
        y = num.array([0, 1, 0, 0])
        width = 300
        height = 100
        config_papersize = (width, height)

        for version in gmtpy.all_installed_gmt_versions():
            gmt = gmtpy.GMT(
                version=version,
                config_papersize=config_papersize)

            for i, cutoff in enumerate([30, 90]):
                gmt.psxy(
                    in_columns=(i*2+x, y),
                    W='10p,red',
                    J='X%gp/%gp' % (width, height),
                    X=0,
                    Y=0,
                    R=(-1, 4, -1, 2),
                    config={
                        'PS_MITER_LIMIT': '%i' % cutoff})

            fname = 'gmtpy_test_override.png'
            fpath = self.fpath(fname)
            gmt.save(fpath)
            self.compare_with_ref(fname, 0.001, show=False)


if __name__ == "__main__":
    plot = True
    util.setup_logging('test_gmtpy', 'warning')
    unittest.main()
