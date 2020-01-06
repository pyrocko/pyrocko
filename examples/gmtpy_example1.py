# gmtpy's GMT class is used for GMT plotting:

from pyrocko.plot.gmtpy import GMT, cm

# For each graphics file to be produced, create a GMT instance.
# The keyword argument `config`, takes a dict with gmtdefaults
# variables you wish to override.

gmt = GMT(config={
    'MAP_FRAME_TYPE': 'fancy',
    'PS_MEDIA': 'Custom_%ix%i' % (15*cm, 15*cm)})

# Every GMT command is now accessible as a method to the GMT instance:

gmt.pscoast(
    R='5/15/52/58',        # region
    J='B10/55/55/60/10c',  # projection
    B='4g4',               # grid
    D='f',                 # resolution
    S=(114, 159, 207),     # wet fill color
    G=(233, 185, 110),     # dry fill color
    W='thinnest')          # shoreline pen

# The PostScript output of the GMT commands is accumulated in memory,
# until the save method is called:

gmt.save('example1.pdf')  # save() looks at the filename extension
gmt.save('example1.eps')  # to determine what format should be saved.
