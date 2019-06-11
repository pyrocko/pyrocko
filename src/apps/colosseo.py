from __future__ import print_function

import sys
import logging
import os.path as op
from optparse import OptionParser

from pyrocko import util, scenario, guts, gf


logger = logging.getLogger('pyrocko.apps.colosseo')
km = 1000.


def d2u(d):
    return dict((k.replace('-', '_'), v) for (k, v) in d.items())


description = '''Earthquake scenario generator from Pyrocko

Create waveforms, InSAR and GNSS offsets of earthquake scenario on the earth.
'''

subcommand_descriptions = {
    'init': 'initialize a new, blank scenario',
    'fill': 'fill the scenario with modelled data',
    'snuffle': 'open Snuffler to inspect the waveform data',
    'map': 'map the scenario arena'
}

subcommand_usages = {
    'init': 'init <scenario_dir> [lat] [lon] [radius_km]',
    'fill': 'fill <scenario_dir>',
    'snuffle': 'snuffle <scenario_dir>',
    'map': '<scenario_dir>',
}

subcommands = subcommand_descriptions.keys()

program_name = 'colosseo'

usage_tdata = d2u(subcommand_descriptions)
usage_tdata['program_name'] = program_name

usage = '''%(program_name)s <subcommand> [options] [--] <arguments> ...

Subcommands:

    init      %(init)s
    fill      %(fill)s
    snuffle   %(snuffle)s
    map       %(map)s

To get further help and a list of available options for any subcommand run:

    %(program_name)s <subcommand> --help

''' % usage_tdata


def die(message, err='', prelude=''):
    if prelude:
        prelude = prelude + '\n'

    if err:
        err = '\n' + err

    sys.exit('%s%s failed: %s%s' % (prelude, program_name, message, err))


def add_common_options(parser):
    parser.add_option(
        '--loglevel',
        action='store',
        dest='loglevel',
        type='choice',
        choices=('critical', 'error', 'warning', 'info', 'debug'),
        default='info',
        help='set logger level to '
             '"critical", "error", "warning", "info", or "debug". '
             'Default is "%default".')


def process_common_options(options):
    util.setup_logging(program_name, options.loglevel)


def cl_parse(command, args, setup=None, details=None):
    usage = subcommand_usages[command]
    descr = subcommand_descriptions[command]

    if isinstance(usage, str):
        usage = [usage]

    susage = '%s %s' % (program_name, usage[0])
    for s in usage[1:]:
        susage += '\n%s%s %s' % (' '*7, program_name, s)

    description = descr[0].upper() + descr[1:] + '.'

    if details:
        description = description + ' %s' % details

    parser = OptionParser(usage=susage, description=description)

    if setup:
        setup(parser)

    add_common_options(parser)
    (options, args) = parser.parse_args(args)
    process_common_options(options)
    return parser, options, args


def get_scenario_yml(path):
    fn = op.join(path, 'scenario.yml')
    if op.exists(fn):
        return fn
    return False


def command_init(args):

    def setup(parser):
        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing files')

    parser, options, args = cl_parse('init', args, setup=setup)

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    scenario_dims = [52., 5.4, 90.]
    scenario_dims[:len(args[1:])] = map(float, args[1:])

    project_dir = args[0]
    logger.info('Initialising new scenario %s at %.2f, %.2f with radius %d km'
                % tuple([args[0]] + scenario_dims))

    try:
        scenario_dims[2] *= km
        scenario.ScenarioGenerator.initialize(
            project_dir, *scenario_dims, force=options.force)

        gf_stores_path = op.join(project_dir, 'gf_stores')
        util.ensuredir(gf_stores_path)

    except scenario.CannotCreate as e:
        die(str(e) + ' Use --force to override.')

    except scenario.ScenarioError as e:
        die(str(e))


def command_fill(args):

    def setup(parser):
        parser.add_option(
            '--force', dest='force', action='store_true',
            help='overwrite existing files')

    parser, options, args = cl_parse('fill', args, setup=setup)

    if len(args) == 0:
        args.append('.')

    fn = get_scenario_yml(args[0])

    if not fn:
        parser.print_help()
        sys.exit(1)

    project_dir = args[0]

    gf_stores_path = op.join(project_dir, 'gf_stores')

    try:
        engine = get_engine([gf_stores_path])

        sc = guts.load(filename=fn)
        sc.init_modelling(engine)
        sc.ensure_gfstores(interactive=True)
        sc.dump_data(
            path=project_dir, overwrite=options.force)
        sc.make_map(op.join(project_dir, 'map.pdf'))

    except scenario.CannotCreate as e:
        die(str(e) + ' Use --force to override.')

    except scenario.ScenarioError as e:
        die(str(e))


def command_map(args):
    parser, options, args = cl_parse('map', args)

    if len(args) == 0:
        args.append('.')

    fn = get_scenario_yml(args[0])

    if not fn:
        parser.print_help()
        sys.exit(1)

    project_dir = args[0]

    gf_stores_path = op.join(project_dir, 'gf_stores')
    engine = get_engine([gf_stores_path])

    try:
        sc = guts.load(filename=fn)
        sc.init_modelling(engine)
        sc.make_map(op.join(project_dir, 'map.pdf'))

    except scenario.ScenarioError as e:
        die(str(e))


def command_snuffle(args):
    from pyrocko.gui import snuffler
    parser, options, args = cl_parse('map', args)

    if len(args) == 0:
        args.append('.')

    fn = get_scenario_yml(args[0])

    if not fn:
        parser.print_help()
        sys.exit(1)

    project_dir = args[0]
    gf_stores_path = op.join(project_dir, 'gf_stores')

    engine = get_engine([gf_stores_path])
    sc = guts.load(filename=fn)
    sc.init_modelling(engine)

    return snuffler.snuffle(
        sc.get_pile(),
        stations=sc.get_stations(),
        events=sc.get_events())


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        sys.exit('Usage: %s' % usage)

    command = args.pop(0)

    if command in subcommands:
        globals()['command_' + command](args)

    elif command in ('--help', '-h', 'help'):
        if command == 'help' and args:
            acommand = args[0]
            if acommand in subcommands:
                globals()['command_' + acommand](['--help'])

        sys.exit('Usage: %s' % usage)

    else:
        sys.exit('%s: error: no such subcommand: %s' % (program_name, command))


def get_engine(gf_store_superdirs):
    engine = gf.LocalEngine(store_superdirs=gf_store_superdirs, use_config=True)
    logger.info(
        'Directories to be searched for GF stores:\n%s'
        % '\n'.join('  ' + s for s in engine.store_superdirs))

    return engine

if __name__ == '__main__':
    main()
