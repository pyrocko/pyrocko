import os
from os.path import join as pjoin
import shutil
import random
import math
import numpy as num
from pyrocko.plot.gmtpy import GMT, inch, cm, golden_ratio, Ax, ScaleGuru, \
    GridLayout, FrameLayout, check_have_gmt, is_gmt5


check_have_gmt()

examples_dir = 'gmtpy_module_examples'
if os.path.exists(examples_dir):
    shutil.rmtree(examples_dir)

os.mkdir(examples_dir)

# Example 1

gmt = GMT()
gmt.pscoast(R='g', J='E32/30/170/8i', B='10g10', D='c', A=10000,
            S=(114, 159, 207), G=(233, 185, 110), W='thinnest')
gmt.save(pjoin(examples_dir, 'example1.pdf'))
gmt.save(pjoin(examples_dir, 'example1.ps'))

# Example 2

gmt = GMT(config_papersize=(7*inch, 7*inch))

gmt.pscoast(
    R='g',
    J='E%g/%g/%g/%gi' % (0., 0., 160., 6.),
    B='0g0',
    D='c',
    A=10000,
    S=(114, 159, 207),
    G=(233, 185, 110),
    W='thinnest',
    X='c',
    Y='c')

rows = []
for i in range(5):
    strike = random.random() * 360.
    dip = random.random() * 90.
    rake = random.random() * 360.-180.
    lat = random.random() * 180.-90.
    lon = random.random() * 360.-180.
    rows.append([lon, lat, 0., strike, dip, rake, 4., 0., 0.,
                 '%.3g, %.3g, %.3g' % (strike, dip, rake)])
gmt.psmeca(
    R=True,
    J=True,
    S='a0.5',
    in_rows=rows)

gmt.save(pjoin(examples_dir, 'example2.ps'))
gmt.save(pjoin(examples_dir, 'example2.pdf'))

# Example 3

if is_gmt5():
    conf = {'PS_PAGE_COLOR': '0/0/0', 'MAP_DEFAULT_PEN': '255/255/255'}
else:
    conf = {'PAGE_COLOR': '0/0/0', 'BASEMAP_FRAME_RGB': '255/255/255'}
gmt = GMT(config=conf)
widget = gmt.default_layout().get_widget()

if is_gmt5():
    gmt.psbasemap(
        R=(-5, 5, -5, 5),
        J='X%gi/%gi' % (5, 5),
        B='1:Time [s]:/1:Amplitude [m]:WSen')
else:
    gmt.psbasemap(
        R=(-5, 5, -5, 5),
        J='X%gi/%gi' % (5, 5),
        B='1:Time [s]:/1:Amplitude [m]:WSen',
        G='100/100/100')

rows = []
for i in range(11):
    rows.append((i-5., random.random()*10.-5.))

gmt.psxy(in_rows=rows, R=True, J=True)
gmt.save(pjoin(examples_dir, 'example3.pdf'))

# Example 4

x = num.linspace(0., math.pi*6, 1001)
y1 = num.sin(x) * 1e-9
y2 = 2.0 * num.cos(x) * 1e-9

xax = Ax(label='Time', unit='s')
yax = Ax(label='Amplitude', unit='m', scaled_unit='nm',
         scaled_unit_factor=1e9, approx_ticks=5, space=0.05)

guru = ScaleGuru([(x, y1), (x, y2)], axes=(xax, yax))
gmt = GMT(config_papersize=(8*inch, 3*inch))
layout = gmt.default_layout()
widget = layout.get_widget()

gmt.draw_layout(layout)

gmt.psbasemap(*(widget.JXY() + guru.RB(ax_projection=True)))
gmt.psxy(in_columns=(x, y1), *(widget.JXY() + guru.R()))
gmt.psxy(in_columns=(x, y2), *(widget.JXY() + guru.R()))
gmt.save(pjoin(examples_dir, 'example4.pdf'), bbox=layout.bbox())
gmt.save(pjoin(examples_dir, 'example4.ps'), bbox=layout.bbox())

# Example 5

x = num.linspace(0., 1e9, 1001)
y = num.sin(x)

axx = Ax(label='Time', unit='s')
ayy = Ax(label='Amplitude', scaled_unit='cm', scaled_unit_factor=100.,
         space=0.05, approx_ticks=5)

guru = ScaleGuru([(x, y)], axes=(axx, ayy))

gmt = GMT(config=conf)
layout = gmt.default_layout()
widget = layout.get_widget()
gmt.psbasemap(*(widget.JXY() + guru.RB(ax_projection=True)))
gmt.psxy(in_columns=(x, y), *(widget.JXY() + guru.R()))
gmt.save(pjoin(examples_dir, 'example5.pdf'), bbox=layout.bbox())

# Example 6

gmt = GMT(config_papersize='a3')

nx, ny = 2, 5
grid = GridLayout(nx, ny)

layout = gmt.default_layout()
layout.set_widget('center', grid)

widgets = []
for iy in range(ny):
    for ix in range(nx):
        inner = FrameLayout()
        inner.set_fixed_margins(1.*cm*golden_ratio, 1.*cm*golden_ratio,
                                1.*cm, 1.*cm)
        grid.set_widget(ix, iy, inner)
        inner.set_vertical(0, (iy+1.))
        widgets.append(inner.get_widget('center'))

gmt.draw_layout(layout)
for widget in widgets:
    x = num.linspace(0., 10., 5)
    y = num.random.rand(5)
    xax = Ax(approx_ticks=4, snap=True)
    yax = Ax(approx_ticks=4, snap=True)
    guru = ScaleGuru([(x, y)], axes=(xax, yax))
    gmt.psbasemap(*(widget.JXY() + guru.RB(ax_projection=True)))
    gmt.psxy(in_columns=(x, y), *(widget.JXY() + guru.R()))

gmt.save(pjoin(examples_dir, 'example6.pdf'), bbox=layout.bbox())
