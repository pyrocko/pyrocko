#!/usr/bin/env python

import sys
import logging
from pyrocko import automap, util, model

from optparse import OptionParser

km = 1000.
logger = logging.getLogger('automap')

program_name = 'automap'

usage = '''
usage: %s [options] [--] <lat> <lon> <radius_km> <output.(pdf|png)>
       %s [--download-etopo1] [--download-srtmgl3]
'''.strip() % (program_name, program_name)

description = '''Create a simple map with topography.'''


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = OptionParser(
        usage=usage,
        description=description)

    parser.add_option(
        '--width',
        dest='width',
        type='float',
        default=20.0,
        metavar='FLOAT',
        help='set width of output image [cm] (%default)')

    parser.add_option(
        '--height',
        dest='height',
        type='float',
        default=15.0,
        metavar='FLOAT',
        help='set height of output image [cm] (%default)')

    parser.add_option(
        '--topo-resolution-min',
        dest='topo_resolution_min',
        type='float',
        default=40.0,
        metavar='FLOAT',
        help='minimum resolution of topography [dpi] (%default)')

    parser.add_option(
        '--topo-resolution-max',
        dest='topo_resolution_max',
        type='float',
        default=200.0,
        metavar='FLOAT',
        help='maximum resolution of topography [dpi] (%default)')

    parser.add_option(
        '--no-grid',
        dest='show_grid',
        default=True,
        action='store_false',
        help='don\'t show grid lines')

    parser.add_option(
        '--no-topo',
        dest='show_topo',
        default=True,
        action='store_false',
        help='don\'t show topography')

    parser.add_option(
        '--no-illuminate',
        dest='illuminate',
        default=True,
        action='store_false',
        help='deactivate artificial illumination of topography')

    parser.add_option(
        '--illuminate-factor-land',
        dest='illuminate_factor_land',
        default='0.5',
        type='float',
        metavar='FLOAT',
        help='set factor for artificial illumination of land (%default)')

    parser.add_option(
        '--illuminate-factor-ocean',
        dest='illuminate_factor_ocean',
        default='0.25',
        type='float',
        metavar='FLOAT',
        help='set factor for artificial illumination of ocean (%default)')

    parser.add_option(
        '--download-etopo1',
        dest='download_etopo1',
        action='store_true',
        help='download full ETOPO1 topography dataset')

    parser.add_option(
        '--download-srtmgl3',
        dest='download_srtmgl3',
        action='store_true',
        help='download full SRTMGL3 topography dataset')

    parser.add_option(
        '--make-decimated-topo',
        dest='make_decimated',
        action='store_true',
        help='pre-make all decimated topography datasets')

    parser.add_option(
        '--stations',
        dest='stations_fn',
        metavar='FILENAME',
        help='load station coordinates from FILENAME')

    parser.add_option(
        '--events',
        dest='events_fn',
        metavar='FILENAME',
        help='load event coordinates from FILENAME')

    parser.add_option(
        '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='print debugging information to stderr')

    (options, args) = parser.parse_args(args)

    if options.debug:
        util.setup_logging(program_name, 'debug')
    else:
        util.setup_logging(program_name, 'info')

    if options.download_etopo1:
        import pyrocko.topo.etopo1
        pyrocko.topo.etopo1.download()

    if options.download_srtmgl3:
        import pyrocko.topo.srtmgl3
        pyrocko.topo.srtmgl3.download()

    if options.make_decimated:
        import pyrocko.topo
        pyrocko.topo.make_all_missing_decimated()

    if (options.download_etopo1 or options.download_srtmgl3 or
            options.make_decimated) and len(args) == 0:

        sys.exit(0)

    if len(args) != 4:
        parser.print_help()
        sys.exit(1)

    try:
        lat = float(args[0])
        lon = float(args[1])
        radius = float(args[2])*km
    except:
        parser.print_help()
        sys.exit(1)

    map = automap.Map(
        width=options.width,
        height=options.height,
        lat=lat,
        lon=lon,
        radius=radius,
        topo_resolution_max=options.topo_resolution_max,
        topo_resolution_min=options.topo_resolution_min,
        show_topo=options.show_topo,
        show_grid=options.show_grid,
        illuminate=options.illuminate,
        illuminate_factor_land=options.illuminate_factor_land,
        illuminate_factor_ocean=options.illuminate_factor_ocean)

    logger.debug('map configuration:\n%s' % str(map))
    map.draw_cities()

    if options.stations_fn:
        stations = model.load_stations(options.stations_fn)
        lats = [s.lat for s in stations]
        lons = [s.lon for s in stations]

        map.gmt.psxy(
            in_columns=(lons, lats),
            S='t8p',
            G='black',
            *map.jxyr)

        for s in stations:
            map.add_label(s.lat, s.lon, '%s.%s' % (s.network, s.station))

    if options.events_fn:
        events = model.load_events(options.events_fn)
        lats = [e.lat for e in events]
        lons = [e.lon for e in events]

        map.gmt.psxy(
            in_columns=(lons, lats),
            S='c8p',
            G='black',
            *map.jxyr)

    map.save(args[3])


if __name__ == '__main__':
    main(sys.argv[1:])
