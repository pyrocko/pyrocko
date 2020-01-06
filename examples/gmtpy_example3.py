from pyrocko.plot.gmtpy import GMT, cm
import numpy as np

# Some data to plot...
x = np.linspace(0, 5, 101)
y = np.sin(x) + 2.5

gmt = GMT(config={
    'PS_PAGE_COLOR': '247/247/240',
    'PS_MEDIA': 'Custom_%ix%i' % (15*cm, 8*cm)})

# Get a default layout for plotting.
# This produces a FrameLayout, a layout built of five widgets,
# a 'center' widget, surrounded by four widgets for the margins:
#
#          +---------------------------+
#          |             top           |
#          +---------------------------+
#          |      |            |       |
#          | left |   center   | right |
#          |      |            |       |
#          +---------------------------+
#          |           bottom          |
#          +---------------------------+

layout = gmt.default_layout()

# We will plot in the 'center' widget:
plot_widget = layout.get_widget('center')

# Set width of plot area to 8 cm and height of the 'top' margin
# to 1 cm. The other values are calculated automatically.
plot_widget.set_horizontal(8*cm)
layout.get_widget('top').set_vertical(1*cm)

# Define how the widget's output parameters are translated
# into -X, -Y and -J option arguments. (This could be skipped
# in this example, because the following templates
# are just the built-in defaults)
plot_widget['X'] = '-Xa%(xoffset)gp'
plot_widget['Y'] = '-Ya%(yoffset)gp'
plot_widget['J'] = '-JX%(width)gp/%(height)gp'

gmt.psbasemap(
    R=(0, 5, 0, 5),
    B='%g:Time [ s ]:/%g:Amplitude [ m ]:SWne' % (1, 1),
    *plot_widget.XYJ())

gmt.psxy(
    R=True,
    W='2p,blue,dotted',
    in_columns=(x, y),
    *plot_widget.XYJ())

# Save the output
gmt.save('example3.pdf')
