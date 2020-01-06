from pyrocko.plot.gmtpy import GMT, cm, GridLayout, FrameLayout, golden_ratio
import numpy as np

# some data to plot...
x = np.linspace(0, 5, 101)
ys = (np.sin(x) + 2.5,  np.cos(x) + 2.5)

gmt = GMT(config={'PS_PAGE_COLOR': '247/247/240'})

layout = GridLayout(1, 2)

widgets = []
for iwidget in range(2):
    inner_layout = FrameLayout()
    layout.set_widget(0, iwidget, inner_layout)
    widget = inner_layout.get_widget('center')
    widget.set_horizontal(7*cm)
    widget.set_vertical(7*cm/golden_ratio)
    widgets.append(widget)

# gmt.draw_layout(layout)
# print(layout)

for widget, y in zip(widgets, ys):
    gmt.psbasemap(
        R=(0, 5, 0, 5),
        B='%g:Time [ s ]:/%g:Amplitude [ m ]:SWne' % (1, 1),
        *widget.XYJ())

    gmt.psxy(
        R=True,
        W='2p,blue,dotted',
        in_columns=(x, y),
        *widget.XYJ())

gmt.save('example4.pdf', bbox=layout.bbox())
