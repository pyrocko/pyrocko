# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Fomosto - Green's function database management.
'''

import sys
import re
import os
import os.path as op
import logging
import copy
import shutil
import argparse

from pyrocko import util, trace, gf, io, fomosto
from pyrocko.gui.snuffler import marker
from pyrocko.util import mpl_show
from pyrocko.cli import PyrockoArgumentParser, dq, ldq

logger = logging.getLogger('pyrocko.apps.fomosto')
km = 1e3

program_name = 'fomosto'

description = '''
Fomosto - Green's function database management.

This is ```fomosto```, the Pyrocko tool to create, build, inspect and manage
Green's function (GF) stores. Fomosto does not compute Green's functions
itself, it provides a unified interface to configure and run several
modelling codes.

This tool's functionality is available through several subcommands. Run
```fomosto SUBCOMMAND --help``` to get further help.
'''

epilog = '''


----

Manual: https://pyrocko.org/docs/current/apps/fomosto

Tutorial: https://pyrocko.org/docs/current/apps/fomosto/tutorial.html
'''


class FomostoError(Exception):
    pass


nice_fatal_errors = (
    FomostoError,
    gf.StoreError,
    gf.meta.GridSpecError,
    gf.meta.OutOfBounds,
    util.DownloadError)


class FomostoArgumentParser(PyrockoArgumentParser):
    '''
    Argument parser with the general options of all fomosto subcommands.
    '''

    def __init__(self, *args, general_options=True, **kwargs):
        kwargs['add_help'] = False
        PyrockoArgumentParser.__init__(self, *args, **kwargs)

        group = self.add_argument_group('General options')
        group.add_argument(
            '--help', '-h',
            action='help',
            help='Show this help message and exit.')

        if general_options:
            loglevel_choices = [
                'critical', 'error', 'warning', 'info', 'debug']
            loglevel_default = 'info'

            group.add_argument(
                '--loglevel',
                choices=loglevel_choices,
                default=loglevel_default,
                metavar='LEVEL',
                help='Set logger level. Choices: %s. Default: %s.' % (
                    ldq(loglevel_choices), dq(loglevel_default)))


def add_store_dir_argument(parser):
    parser.add_argument(
        'store_dir',
        nargs='?',
        default=op.curdir,
        metavar='STORE_DIR',
        help='Store directory. Default: current directory.')


def add_store_dirs_argument(parser):
    parser.add_argument(
        'store_dirs',
        nargs='*',
        default=[op.curdir],
        metavar='STORE_DIR',
        help='Store directories. Default: current directory.')


def add_force_argument(parser):
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files.')


def check_store_dir(store_dir):
    store_dir = op.abspath(store_dir)
    if not op.isdir(store_dir):
        raise FomostoError('Not a directory: %s' % store_dir)

    return store_dir


def fomo_wrapper_module(name):
    if not re.match(gf.meta.StringID.pattern, name):
        raise FomostoError('Invalid modelling code ID: %s' % name)

    words = name.split('.', 1)
    if len(words) == 2:
        name, variant = words
    else:
        name = words[0]
        variant = None

    name_clean = re.sub(r'[.-]', '_', name)
    modname = '.'.join(['pyrocko', 'fomosto', name_clean])
    try:
        mod = __import__(modname, level=0)
    except ImportError:
        raise FomostoError(
            'Modelling code wrapper "%s" not available or not installed '
            '(module probed: "%s").' % (name, modname))

    return getattr(mod.fomosto, name_clean), variant


def backends_details():
    # The docs should not show what is installed on the machine building them.
    for_docs = os.environ.get('PYROCKO_RST_HELP', '0') != '0'

    lines = []
    for modelling_code_id, missing in fomosto.get_backend_variants():

        if for_docs:
            lines.append('    %s' % modelling_code_id)
            continue

        if missing is None:
            status = 'not usable, cannot import backend module'
        elif missing:
            status = 'not installed, missing: %s' % ', '.join(missing)
        else:
            status = 'installed'

        lines.append('    %-25s %s' % (modelling_code_id, status))

    return '\n'.join(lines)


def sindex(args):
    return '(%s)' % ', '.join('%g' % x for x in args)


def load_config(fn):
    try:
        config = gf.meta.load(filename=fn)
        assert isinstance(config, gf.Config)

    except Exception:
        raise FomostoError('Cannot load GF config from file: %s' % fn)

    return config


def setup_init(parser):
    parser.add_argument(
        'modelling_code_id',
        metavar='MODELLING_CODE_ID',
        help='Modelling code to use, including its variant, e.g. '
             '```qssp.2017```, or ```redeploy``` to create a store with the '
             'configuration of an existing one.')

    parser.add_argument(
        'store_dir',
        metavar='STORE_DIR',
        help='Directory of the new store, or of the existing store to take '
             'the configuration from with ```redeploy```.')

    parser.add_argument(
        'dest_dir',
        nargs='?',
        metavar='DESTINATION',
        help='Directory of the new store (only with ```redeploy```).')


def run_init(parser, args):
    if args.modelling_code_id == 'redeploy':
        if args.dest_dir is None:
            parser.error('A destination is required with "redeploy".')

        source_dir, dest_dir = args.store_dir, args.dest_dir

        source = gf.Store(source_dir)

        config = copy.deepcopy(source.config)
        config.derived_from_id = source.config.id
        config_filenames = gf.store.Store.create_editables(
            dest_dir, config=config)

        dest = gf.Store(dest_dir)

        for k in source.extra_keys():
            source_fn = source.get_extra_path(k)
            dest_fn = dest.get_extra_path(k)
            shutil.copyfile(source_fn, dest_fn)

        logger.info(
            '(1) Configure settings in files:\n  %s'
            % '\n  '.join(config_filenames))

        logger.info(
            '(2) Run "fomosto redeploy <source> <dest>", as needed.')

    else:
        if args.dest_dir is not None:
            parser.error('Too many arguments.')

        store_dir = args.store_dir

        module, variant = fomo_wrapper_module(args.modelling_code_id)
        config_filenames = module.init(store_dir, variant)

        logger.info('(1) Configure settings in files:\n  %s'
                    % '\n  '.join(config_filenames))
        logger.info('(2) Run "fomosto ttt" in directory "%s".' % store_dir)
        logger.info('(3) Run "fomosto build" in directory "%s".' % store_dir)


def setup_build(parser):
    add_store_dir_argument(parser)
    add_force_argument(parser)

    parser.add_argument(
        '--nworkers',
        type=int,
        metavar='N',
        help='Run N worker processes in parallel. Default: number of CPUs.')

    parser.add_argument(
        '--continue',
        dest='continue_',
        action='store_true',
        help='Continue suspended build.')

    parser.add_argument(
        '--step',
        type=int,
        metavar='I',
        help='Process step number I only.')

    parser.add_argument(
        '--block',
        dest='iblock',
        type=int,
        metavar='I',
        help='Process block number I only (requires ``--step`` for '
             'backends with several steps). With ```--block=0```, print '
             'the commands to process all blocks separately.')


def run_build(parser, args):
    store_dir = check_store_dir(args.store_dir)

    step = args.step - 1 if args.step is not None else None
    iblock = args.iblock - 1 if args.iblock is not None else None

    store = gf.Store(store_dir)
    module, _ = fomo_wrapper_module(store.config.modelling_code_id)
    module.build(
        store_dir,
        force=args.force,
        nworkers=args.nworkers,
        continue_=args.continue_,
        step=step,
        iblock=iblock)


def setup_stats(parser):
    add_store_dir_argument(parser)


def run_stats(parser, args):
    store = gf.Store(check_store_dir(args.store_dir))
    s = store.stats()

    for k in store.stats_keys:
        print('%s: %s' % (k, s[k]))


def setup_check(parser):
    add_store_dir_argument(parser)


def run_check(parser, args):
    store_dir = check_store_dir(args.store_dir)
    store = gf.Store(store_dir)
    problems = store.check(show_progress=True)
    if problems:
        raise FomostoError('Problems detected with GF store: %s' % store_dir)


def setup_decimate(parser):
    add_store_dir_argument(parser)

    parser.add_argument(
        'factor',
        type=int,
        metavar='FACTOR',
        help='Decimation factor.')

    parser.add_argument(
        '--config',
        dest='config_fn',
        metavar='FILE',
        help='Use modified spatial sampling given in FILE.')

    add_force_argument(parser)


def run_decimate(parser, args):
    store_dir = check_store_dir(args.store_dir)

    config = None
    if args.config_fn:
        config = load_config(args.config_fn)

    store = gf.Store(store_dir)
    store.make_decimated(
        args.factor, config=config, force=args.force, show_progress=True)


def setup_redeploy(parser):
    parser.add_argument(
        'source_store_dir',
        metavar='SOURCE',
        help='Store to copy the traces from.')

    parser.add_argument(
        'dest_store_dir',
        metavar='DESTINATION',
        help='Store to copy the traces into.')


def run_redeploy(parser, args):
    source = gf.Store(args.source_store_dir)

    try:
        gf.store.Store.create_dependants(args.dest_store_dir)
    except gf.StoreError:
        pass

    dest = gf.Store(args.dest_store_dir, 'w')

    show_progress = True

    try:
        if show_progress:
            pbar = util.progressbar('redeploying', dest.config.nrecords)

        for i, node in enumerate(dest.config.iter_nodes()):
            try:
                tr = source.get(node, interpolation='off')
                dest.put(node, tr)

            except (gf.meta.OutOfBounds, gf.store.NotAllowedToInterpolate) as e:  # noqa
                logger.debug('Skipping %s (%s).' % (sindex(node), e))

            except gf.store.StoreError as e:
                logger.warning(
                    'Cannot insert %s (%s).' % (sindex(node), e))

            if show_progress:
                pbar.update(i+1)

    finally:
        if show_progress:
            pbar.finish()


def setup_trans2rot(parser):
    parser.add_argument(
        'source_store_dir',
        metavar='SOURCE',
        help='Translational input store.')

    parser.add_argument(
        'dest_store_dir',
        metavar='DESTINATION',
        help='Rotational output store.')


def run_trans2rot(parser, args):
    from pyrocko.fomosto.elastic10_to_rotational8 \
        import elastic10_to_rotational8

    elastic10_to_rotational8(args.source_store_dir, args.dest_store_dir)


def setup_fd2trans(parser):
    parser.add_argument(
        'source_store_dir',
        metavar='SOURCE',
        help='Input store with finite difference support (elastic10_fd).')

    parser.add_argument(
        'dest_store_dir',
        metavar='DESTINATION',
        help='Output store (elastic10).')


def run_fd2trans(parser, args):
    from pyrocko.fomosto.elastic10_fd_to_elastic10 \
        import elastic10_fd_to_elastic10

    elastic10_fd_to_elastic10(args.source_store_dir, args.dest_store_dir)


def setup_view(parser):
    add_store_dirs_argument(parser)

    parser.add_argument(
        '--extract',
        metavar='SELECTION',
        help='Specify which traces to show, as ```start:stop[:step|@num],'
             '...```, one range per store dimension.')

    parser.add_argument(
        '--show-phases',
        dest='showphases',
        metavar='PHASE_IDS',
        help='Add phase markers from the travel time tables. Comma-separated '
             'list of phase IDs, or ```all```.')

    parser.add_argument(
        '--opengl',
        dest='opengl',
        action='store_true',
        default=None,
        help='Use OpenGL for drawing.')

    parser.add_argument(
        '--no-opengl',
        dest='opengl',
        action='store_false',
        default=None,
        help='Do not use OpenGL for drawing.')


def run_view(parser, args):
    gdef = None
    if args.extract:
        gdef = gf.meta.parse_grid_spec(args.extract)

    store_dirs = [check_store_dir(x) for x in args.store_dirs]

    alpha = 'abcdefghijklmnopqrstxyz'.upper()

    markers = []
    traces = []

    for istore, store_dir in enumerate(store_dirs):
        store = gf.Store(store_dir)
        if args.showphases == 'all':
            phasenames = [pn.id for pn in store.config.tabulated_phases]
        elif args.showphases is not None:
            phasenames = args.showphases.split(',')

        ii = 0
        for node in store.config.iter_extraction(gdef):
            gtr = store.get(node)

            loc_code = ''
            if len(store_dirs) > 1:
                loc_code = alpha[istore % len(alpha)]

            if gtr:

                sta_code = '%04i (%s)' % (
                    ii, ','.join('%gk' % (x/1000.) for x in node[:-1]))
                tmin = gtr.deltat * gtr.itmin

                tr = trace.Trace(
                    '',
                    sta_code,
                    loc_code,
                    '%02i' % node[-1],
                    ydata=gtr.data,
                    deltat=gtr.deltat,
                    tmin=tmin)

                if args.showphases:
                    for phasename in phasenames:
                        phase_tmin = store.t(phasename, node[:-1])
                        if phase_tmin:
                            m = marker.PhaseMarker(
                                [('',
                                  sta_code,
                                  loc_code,
                                  '%02i' % node[-1])],
                                phase_tmin,
                                phase_tmin,
                                0,
                                phasename=phasename)
                            markers.append(m)

                traces.append(tr)

            ii += 1

    def setup(win):
        viewer = win.get_view()
        viewer.menuitem_showboxes.setChecked(False)
        viewer.menuitem_colortraces.setChecked(True)
        viewer.menuitem_demean.setChecked(False)
        viewer.menuitems_sorting[5][0].setChecked(True)
        viewer.menuitems_scaling[2][0].setChecked(True)
        viewer.sortingmode_change()
        viewer.scalingmode_change()

    trace.snuffle(
        traces, markers=markers, opengl=args.opengl, launch_hook=setup)


def setup_extract(parser):
    add_store_dir_argument(parser)

    parser.add_argument(
        'selection',
        metavar='SELECTION',
        help='Traces to extract, as ```start:stop[:step|@num],...```, one '
             'range per store dimension.')

    format_choices = ['mseed', 'sac', 'text', 'yaff']
    format_default = 'mseed'
    parser.add_argument(
        '--format',
        default=format_default,
        choices=format_choices,
        metavar='FORMAT',
        help='Export to format FORMAT. Choices: %s. Default: %s.' % (
            ldq(format_choices), dq(format_default)))

    fndfl = 'extracted/%(irecord)s_%(args)s.%(extension)s'
    parser.add_argument(
        '--output',
        dest='output_fn',
        default=fndfl,
        metavar='TEMPLATE',
        help='Output path template. Default: ```%s```.'
             % fndfl.replace('%', '%%'))


def run_extract(parser, args):
    gdef = gf.meta.parse_grid_spec(args.selection)

    store_dir = check_store_dir(args.store_dir)

    extensions = {
        'mseed': 'mseed',
        'sac': 'sac',
        'text': 'txt',
        'yaff': 'yaff'}

    store = gf.Store(store_dir)
    for node in store.config.iter_extraction(gdef):
        gtr = store.get(node)
        if gtr:
            tr = trace.Trace(
                '', '', '', util.zfmt(store.config.ncomponents) % node[-1],
                ydata=gtr.data,
                deltat=gtr.deltat,
                tmin=gtr.deltat * gtr.itmin)

            additional = dict(
                args='_'.join('%g' % x for x in node),
                irecord=store.str_irecord(node),
                extension=extensions[args.format])

            io.save(
                tr,
                args.output_fn,
                format=args.format,
                additional=additional)


def setup_import(parser):
    parser.add_argument(
        'source_path',
        metavar='SOURCE',
        help='Kiwi GFDB to import.')

    parser.add_argument(
        'dest_store_dir',
        metavar='DESTINATION',
        help='GF store to create.')


def run_import(parser, args):
    try:
        from tunguska import gfdb
    except ImportError:
        raise FomostoError(
            'The Kiwi tools must be installed to use this feature.')

    show_progress = True

    source_path = args.source_path
    dest_store_dir = args.dest_store_dir

    if op.isdir(source_path):
        source_path = op.join(source_path, 'db')

    source_path = re.sub(r'(\.\d+\.chunk|\.index)$', '', source_path)

    db = gfdb.Gfdb(source_path)

    config = gf.meta.ConfigTypeA(
        id='imported_gfs',
        distance_min=db.firstx,
        distance_max=db.firstx + (db.nx-1) * db.dx,
        distance_delta=db.dx,
        source_depth_min=db.firstz,
        source_depth_max=db.firstz + (db.nz-1) * db.dz,
        source_depth_delta=db.dz,
        sample_rate=1.0/db.dt,
        ncomponents=db.ng
    )

    gf.store.Store.create(dest_store_dir, config=config)
    dest = gf.Store(dest_store_dir, 'w')
    try:
        if show_progress:
            pbar = util.progressbar(
                'importing', dest.config.nrecords/dest.config.ncomponents)

        for i, node in enumerate(dest.config.iter_nodes(level=-1)):
            source_depth, distance = [float(x) for x in node]
            traces = db.get_traces_pyrocko(distance, source_depth)
            ig_to_trace = dict((tr.meta['ig']-1, tr) for tr in traces)
            for ig in range(db.ng):
                if ig in ig_to_trace:
                    tr = ig_to_trace[ig]
                    gf_tr = gf.store.GFTrace(
                        tr.get_ydata(),
                        int(round(tr.tmin / tr.deltat)),
                        tr.deltat)

                else:
                    gf_tr = gf.store.Zero

                dest.put((source_depth, distance, ig), gf_tr)

            if show_progress:
                pbar.update(i+1)

    finally:
        if show_progress:
            pbar.finish()

    dest.close()


def setup_export(parser):
    add_store_dir_argument(parser)

    parser.add_argument(
        'target_path',
        metavar='DESTINATION',
        help='Kiwi GFDB to create. If it is a directory, the GFDB is '
             'created in it, with name ```kiwi_gfdb```.')

    parser.add_argument(
        '--nchunks',
        type=int,
        default=1,
        metavar='N',
        help='Split output GFDB into N chunks. Default: 1.')


def run_export(parser, args):
    from subprocess import Popen, PIPE

    try:
        from tunguska import gfdb
    except ImportError as e:
        raise FomostoError(
            'The Kiwi tools must be installed to use this feature (%s).' % e)

    show_progress = True

    target_path = args.target_path
    if op.isdir(target_path):
        target_path = op.join(target_path, 'kiwi_gfdb')
        logger.warning('Exported GFDB will be named "%s.*".' % target_path)

    source_store_dir = check_store_dir(args.store_dir)

    source = gf.Store(source_store_dir, 'r')
    config = source.config

    if not isinstance(config, gf.meta.ConfigTypeA):
        raise FomostoError(
            'Only stores of type A can be exported to Kiwi format.')

    if op.isfile(target_path + '.index'):
        raise FomostoError('Destination already exists: %s' % target_path)

    cmd = [str(x) for x in [
        'gfdb_build',
        target_path,
        args.nchunks,
        config.ndistances,
        config.nsource_depths,
        config.ncomponents,
        config.deltat,
        config.distance_delta,
        config.source_depth_delta,
        config.distance_min,
        config.source_depth_min]]

    p = Popen(cmd, stdin=PIPE)
    p.communicate()

    out_db = gfdb.Gfdb(target_path)

    try:
        if show_progress:
            pbar = util.progressbar(
                'exporting', config.nrecords/config.ncomponents)

        for i, (z, x) in enumerate(config.iter_nodes(level=-1)):

            data_out = []
            for ig in range(config.ncomponents):
                try:
                    tr = source.get((z, x, ig), interpolation='off')
                    data_out.append((tr.t, tr.data * config.factor))

                except gf.store.StoreError as e:
                    logger.warning(
                        'Cannot get %s (%s).' % (sindex((z, x, ig)), e))
                    data_out.append(None)

            # put a zero valued sample to no-data zero-traces at a compatible
            # time
            tmins = [
                entry[0][0]
                for entry in data_out
                if entry is not None and entry[0].size != 0]

            if tmins:
                tmin = min(tmins)
                for entry in data_out:
                    if entry is not None and entry[0].size == 0:
                        entry[0].resize(1)
                        entry[1].resize(1)
                        entry[0][0] = tmin
                        entry[1][0] = 0.0

            out_db.put_traces_slow(x, z, data_out)

            if show_progress:
                pbar.update(i+1)

        source.close()

    finally:
        if show_progress:
            pbar.finish()


def stored_attribute_table_plots(phase_ids, args, attribute):
    import numpy as num
    from pyrocko.plot.cake_plot import xscaled, yscaled, mpl_init
    from pyrocko.plot import mpl_labelspace as labelspace

    plt = mpl_init()

    np = 1
    store_dir = check_store_dir(args.store_dir)
    for phase_id in phase_ids:
        store = gf.Store(store_dir)

        if args.receiver_depth is not None:
            receiver_depth = args.receiver_depth * 1000.0
        else:
            if isinstance(store.config, (gf.ConfigTypeA, gf.ConfigTypeC)):
                receiver_depth = store.config.receiver_depth

            elif isinstance(store.config, gf.ConfigTypeB):
                receiver_depth = store.config.receiver_depth_min

            else:
                receiver_depth = 0.0

        phase = store.get_stored_phase(phase_id, attribute)
        axes = plt.subplot(2, len(phase_ids), np)
        labelspace(axes)
        xscaled(1. / km, axes)
        yscaled(1. / km, axes)
        x = None
        if isinstance(store.config, gf.ConfigTypeB):
            x = (receiver_depth, None, None)

        phase.plot_2d(axes, x=x)
        axes.set_title(phase_id)
        np += 1

    axes = plt.subplot(2, 1, 2)
    num_d = 100
    distances = num.linspace(store.config.distance_min,
                             store.config.distance_max,
                             num_d)

    if args.source_depth is not None:
        source_depth = args.source_depth * km
    else:
        source_depth = store.config.source_depth_min + (
            store.config.source_depth_max - store.config.source_depth_min) / 2.

    if isinstance(store.config, gf.ConfigTypeA):
        attribute_vals = num.empty(num_d)
        for phase_id in phase_ids:
            attribute_vals[:] = num.nan
            for i, d in enumerate(distances):
                if attribute == 'phase':
                    attribute_vals[i] = store.t(phase_id, (source_depth, d))
                    ylabel = 'TT [s]'
                else:
                    attribute_vals[i] = store.get_stored_attribute(
                        phase_id, args.attribute, (source_depth, d))
                    ylabel = '%s [deg]' % args.attribute

            axes.plot(distances / km, attribute_vals, label=phase_id)

        axes.set_title('source source_depth %s km' % (source_depth / km))
        axes.set_xlabel('distance [km]')
        axes.set_ylabel(ylabel)
        axes.legend()

    plt.tight_layout()
    mpl_show(plt)


def add_attribute_plot_arguments(parser):
    parser.add_argument(
        '--source-depth',
        type=float,
        metavar='FLOAT',
        help='Source depth [km]. Default: centre of the store\'s source '
             'depth range.')

    parser.add_argument(
        '--receiver-depth',
        type=float,
        metavar='FLOAT',
        help='Receiver depth [km]. Default: the store\'s (minimum) '
             'receiver depth.')


def setup_ttt(parser):
    add_store_dir_argument(parser)
    add_force_argument(parser)


def run_ttt(parser, args):
    store = gf.Store(check_store_dir(args.store_dir))
    store.make_travel_time_tables(force=args.force)


def setup_tttview(parser):
    add_store_dir_argument(parser)

    parser.add_argument(
        'phase_ids',
        metavar='PHASE_IDS',
        help='Comma-separated list of phase IDs, e.g. ```Pdiff,S```.')

    add_attribute_plot_arguments(parser)


def run_tttview(parser, args):
    phase_ids = args.phase_ids.split(',')
    stored_attribute_table_plots(phase_ids, args, attribute='phase')


def setup_sat(parser):
    add_store_dir_argument(parser)
    add_force_argument(parser)

    attribute_choices = gf.store.available_stored_tables[1:]
    attribute_default = 'takeoff_angle'
    parser.add_argument(
        '--attribute',
        choices=attribute_choices,
        default=attribute_default,
        metavar='ATTRIBUTE',
        help='Calculate interpolation table for the given ray attribute. '
             'Choices: %s. Default: %s.' % (
                 ldq(attribute_choices), dq(attribute_default)))


def run_sat(parser, args):
    store = gf.Store(check_store_dir(args.store_dir))
    store.make_stored_table(args.attribute, force=args.force)


def setup_satview(parser):
    add_store_dir_argument(parser)

    parser.add_argument(
        'phase_ids',
        metavar='PHASE_IDS',
        help='Comma-separated list of phase IDs, e.g. ```Pdiff,S```.')

    add_attribute_plot_arguments(parser)

    attribute_choices = gf.store.available_stored_tables[1:]
    attribute_default = 'takeoff_angle'
    parser.add_argument(
        '--attribute',
        choices=attribute_choices,
        default=attribute_default,
        metavar='ATTRIBUTE',
        help='View the given ray attribute. Choices: %s. Default: %s.' % (
            ldq(attribute_choices), dq(attribute_default)))


def run_satview(parser, args):
    phase_ids = args.phase_ids.split(',')

    logger.info('Plotting stored attribute "%s".' % args.attribute)

    stored_attribute_table_plots(phase_ids, args, attribute=args.attribute)


def setup_tttextract(parser):
    add_store_dir_argument(parser)

    parser.add_argument(
        'phases',
        metavar='PHASES',
        help='Comma-separated list of phase IDs or timing definitions.')

    parser.add_argument(
        'selection',
        metavar='SELECTION',
        help='Source/receiver configurations, as '
             '```start:stop[:step|@num],...```, one range per store '
             'dimension (except the component dimension).')

    parser.add_argument(
        '--output',
        dest='output_fn',
        metavar='TEMPLATE',
        help='Output to text files instead of standard output, e.g. '
             '```extracted/%%(args)s.txt```.')


def run_tttextract(parser, args):
    try:
        phases = [gf.meta.Timing(x.strip()) for x in args.phases.split(',')]
    except gf.meta.InvalidTimingSpecification:
        parser.error('Invalid phase specification: "%s".' % args.phases)

    gdef = gf.meta.parse_grid_spec(args.selection)

    store_dir = check_store_dir(args.store_dir)

    store = gf.Store(store_dir)
    for node in store.config.iter_extraction(gdef, level=-1):
        s = ['%e' % x for x in node]
        for phase in phases:
            t = store.t(phase, node)
            if t is not None:
                s.append('%e' % t)
            else:
                s.append('nan')

        if args.output_fn:
            d = dict(
                args='_'.join('%e' % x for x in node),
                extension='txt')

            fn = args.output_fn % d
            util.ensuredirs(fn)
            with open(fn, 'a') as f:
                f.write(' '.join(s))
                f.write('\n')
        else:
            print(' '.join(s))


def setup_tttlsd(parser):
    add_store_dir_argument(parser)

    parser.add_argument(
        'phase_ids',
        metavar='PHASE_IDS',
        help='Comma-separated list of phase IDs.')


def run_tttlsd(parser, args):
    phase_ids = [x.strip() for x in args.phase_ids.split(',')]

    store_dir = check_store_dir(args.store_dir)

    store = gf.Store(store_dir)
    for phase_id in phase_ids:
        store.fix_ttt_holes(phase_id)


def setup_server(parser):
    from pyrocko import server

    parser.add_argument(
        'store_superdirs',
        nargs='+',
        metavar='STORE_SUPER_DIR',
        help='Directories containing the stores to serve.')

    server.add_cli_arguments(
        parser,
        exclude=('page', 'open', 'cookie_secret_path'),
        default_port=8080)


def run_server(parser, args):
    from pyrocko.gf import server

    engine = gf.LocalEngine(store_superdirs=args.store_superdirs)

    server.run(
        engine,
        host=args.host,
        port=args.port,
        debug=args.debug)


def setup_download(parser):
    parser.add_argument(
        'site',
        metavar='SITE',
        help='Server to download from, e.g. ```kinherd```.')

    parser.add_argument(
        'store_id',
        nargs='?',
        metavar='STORE_ID',
        help='ID of the store to download. If not given, list the stores '
             'available on the server.')

    add_force_argument(parser)


def run_download(parser, args):
    from pyrocko.gf import ws

    site, store_id = args.site, args.store_id

    if store_id is not None \
            and not re.match(gf.meta.StringID.pattern, store_id):
        raise FomostoError('Invalid store ID: %s' % store_id)

    if site not in gf.ws.g_site_abbr:
        if -1 == site.find('://'):
            site = 'http://' + site

    ws.download_gf_store(site=site, store_id=store_id, force=args.force)


def setup_modelview(parser):
    add_store_dirs_argument(parser)

    parser.add_argument(
        '--parameters',
        default='vp,vs',
        metavar='PARAMETERS',
        help='Comma-separated list of the parameters to plot. Choices: '
             '```vp```, ```vs```, ```rho```, ```qp```, ```qs```, '
             '```vp/vs```, ```qp/qs```. Default: ```vp,vs```.')


def run_modelview(parser, args):

    import matplotlib.pyplot as plt
    import numpy as num
    from pyrocko.plot.cake_plot import mpl_init, xscaled, yscaled
    from pyrocko.plot import mpl_labelspace as labelspace
    mpl_init()

    neat_labels = {
        'vp': '$v_p$',
        'vs': '$v_s$',
        'qp': '$Q_p$',
        'qs': '$Q_s$',
        'rho': '$\\rho$'}

    units = {'vp': '[km/s]', 'vs': '[km/s]', 'rho': '[g/cm^3]'}

    store_dirs = [check_store_dir(x) for x in args.store_dirs]

    parameters = args.parameters.split(',')

    fig, axes = plt.subplots(1,
                             len(parameters),
                             sharey=True,
                             figsize=(len(parameters)*3, 5))

    if not isinstance(axes, num.ndarray):
        axes = [axes]

    axes = dict(zip(parameters, axes))

    for store_id in store_dirs:
        store = gf.Store(store_id)
        mod = store.config.earthmodel_1d
        z = mod.profile('z')

        for p in parameters:
            ax = axes[p]
            labelspace(ax)
            if '/' in p:
                p1, p2 = p.split('/')
                profile = mod.profile(p1)/mod.profile(p2)
            else:
                profile = mod.profile(p)

            ax.plot(profile, -z, label=store_id.split('/')[-1])
            if p in ['vs', 'vp', 'rho']:
                xscaled(1./1000., ax)

            yscaled(1./km, ax)

    for p, ax in axes.items():
        ax.grid()
        if p in neat_labels:
            lab = neat_labels[p]
        elif all(x in neat_labels for x in p.split('/')):
            lab = '/'.join(neat_labels[x] for x in p.split('/'))
        else:
            lab = p

        ax.set_title(lab, y=1.02)

        if p in units:
            lab += ' ' + units[p]

        ax.autoscale()
        ax.set_xlabel(lab)

    axes[parameters[0]].set_ylabel('Depth [km]')

    handles, labels = ax.get_legend_handles_labels()

    if len(store_dirs) > 1:
        ax.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, 0.12),
            bbox_transform=fig.transFigure,
            ncol=3,
            loc='upper center',
            fancybox=True)

    plt.subplots_adjust(bottom=0.22,
                        wspace=0.05)

    mpl_show(plt)


def setup_upgrade(parser):
    add_store_dirs_argument(parser)


def run_upgrade(parser, args):
    store_dirs = [check_store_dir(x) for x in args.store_dirs]
    for store_dir in store_dirs:
        store = gf.Store(store_dir)
        nup = store.upgrade()
        if nup == 0:
            print('%s: already up-to-date.' % store_dir)
        else:
            print('%s: %i file%s upgraded.' % (
                store_dir, nup, ['s', ''][nup == 1]))


def setup_addref(parser):
    parser.add_argument(
        'paths',
        nargs='+',
        metavar='PATH',
        help='Store directories (default: current directory) and BibTeX '
             'files with the references to add.')


def run_addref(parser, args):
    store_dirs = []
    filenames = []
    for path in args.paths:
        if op.isdir(path):
            store_dirs.append(path)
        elif op.isfile(path):
            filenames.append(path)
        else:
            raise FomostoError('Invalid argument: %s' % path)

    if not store_dirs:
        store_dirs.append('.')

    references = []
    try:
        for filename in filenames:
            references.extend(gf.meta.Reference.from_bibtex(filename=filename))
    except ImportError:
        raise FomostoError(
            'The pybtex module must be installed to use this function.')

    if not references:
        raise FomostoError('No references found.')

    for store_dir in store_dirs:
        store = gf.Store(store_dir)
        ids = [ref.id for ref in store.config.references]
        for ref in references:
            if ref.id in ids:
                raise FomostoError('Duplicate reference ID: %s' % ref.id)

            ids.append(ref.id)
            store.config.references.append(ref)

        store.save_config(make_backup=True)


def setup_qc(parser):
    add_store_dir_argument(parser)


def run_qc(parser, args):
    store = gf.Store(check_store_dir(args.store_dir))
    s = store.stats()
    if s['empty'] != 0:
        print('has empty records')

    for aname in ['author', 'author_email', 'description', 'references']:

        if not getattr(store.config, aname):
            print('%s empty' % aname)


def setup_report(parser):
    parser.add_argument(
        'report_args',
        nargs=argparse.REMAINDER,
        metavar='...',
        help='Subcommand and arguments of the report tool. Run '
             '```fomosto report``` without arguments to list its '
             'subcommands.')


def run_report(parser, args):
    from pyrocko.fomosto.report import report_call
    report_call.run_program(args.report_args)


subcommands = [
    ('init', 'Create a new empty GF store.', '''
The new store is configured for the given modelling code, which must include
the variant of the backend, e.g. ```qssp.2017```.

With ```fomosto init redeploy SOURCE DESTINATION```, a new store is created
with the configuration of an existing store.

The following modelling backends and variants are available.

%s

Installation instructions and more information at
https://pyrocko.org/docs/current/apps/fomosto/backends.html
'''),
    ('build', 'Compute GFs and fill into store.', None),
    ('stats', 'Print information about a GF store.', None),
    ('check', 'Check for problems in GF store.', None),
    ('decimate', 'Build decimated variant of a GF store.', None),
    ('redeploy', 'Copy traces from one GF store into another.', None),
    ('view', 'View selected traces.', None),
    ('extract', 'Extract selected traces.', None),
    ('import', 'Convert Kiwi GFDB to GF store format.', None),
    ('export', 'Convert GF store to Kiwi GFDB format.', None),
    ('ttt', 'Create travel time tables.', None),
    ('tttview', 'Plot travel time table.', None),
    ('tttextract', 'Extract selected travel times.', None),
    ('tttlsd', 'Fix holes in travel time tables.', '''
This subcommand fills holes in travel time tables by using an eikonal solver
to predict travel times for the missing values. The approach works best for
simple P or S phases without any reflections or conversions. It creates new
travel time tables which are named by adding the suffix ```.lsd``` to the
original name. E.g. running ```fomosto tttlsd begin``` would produce a new
travel time table ```begin.lsd```. The new name can be referenced anywhere
where a stored travel time table can be used, e.g. in ```extra/qseis```.
'''),
    ('sat', 'Create stored ray attribute table.', None),
    ('satview', 'Plot stored ray attribute table.', None),
    ('server', 'Run seismosizer server.', None),
    ('download', 'Download GF store from a server.', '''
Browse pre-calculated Green's function stores online:
https://greens-mill.pyrocko.org
'''),
    ('modelview', 'Plot earth models.', None),
    ('upgrade', 'Upgrade store format to latest version.', None),
    ('addref', 'Import citation references to GF store config.', None),
    ('qc', 'Check GF store for missing data and metadata.', None),
    ('report', 'Create reports for Green\'s function databases.', None),
    ('trans2rot', 'Create rotational from translational store.', '''
Create a rotational GF store from a translational one using finite
differences approximations (elastic10 or elastic10_fd to rotational8).

If the source store is of type A and receivers at the surface, the free
surface condition is used to replace the vertical derivatives with horizontal
ones. If the input is of type B, the complete set of derivatives are used. It
is important that the spatial sampling of the source store is high enough so
that the finite difference approximations are sufficiently accurate.

If the destination store directory does not exist, it is created and the
extent and spacing of the source store is used for the rotational store. If it
does exist and contains a valid ```config```, it is used in place in order to
allow for different input and output extents and spacings. The store must be
empty in the latter case (delete the ```index``` and ```traces``` files if
needed).
'''),
    ('fd2trans', 'Extract elastic10 from elastic10_fd store.', '''
Extract regular GF store (elastic10) from a finite difference support GF store
(elastic10_fd).
'''),
]


def make_parser():
    parser = FomostoArgumentParser(
        prog=program_name,
        description=description,
        epilog=epilog,
        general_options=False)

    subparsers = parser.add_subparsers(
        dest='subcommand',
        metavar='SUBCOMMAND',
        title='Subcommands')

    for name, headline, details in subcommands:
        subdescription = headline
        if details:
            if name == 'init':
                details = details % backends_details()

            subdescription += '\n' + details

        subparser = subparsers.add_parser(
            name,
            help=headline,
            description=subdescription,
            epilog=epilog,
            # the report tool parses its arguments itself
            general_options=name != 'report')

        globals()['setup_' + name](subparser)
        subparser.set_defaults(
            target=globals()['run_' + name], subparser=subparser)

    return parser


def main(args=None):
    '''
    CLI entry point for Pyrocko's ``fomosto`` app.
    '''

    if args is None:
        args = sys.argv[1:]

    args = list(args)

    if args and args[0] == 'help':
        args = args[1:2] + ['--help']

    if args and args[0] == 'report' and args[1:] not in (['--help'], ['-h']):
        # The report tool parses its arguments itself.
        util.setup_logging(program_name, 'info')
        from pyrocko.fomosto.report import report_call
        report_call.run_program(args[1:])
        return

    parser = make_parser()
    args = parser.parse_args(args)

    if 'target' not in args:
        parser.print_help()
        return

    util.setup_logging(program_name, getattr(args, 'loglevel', 'info'))

    try:
        args.target(args.subparser, args)

    except nice_fatal_errors as e:
        logger.fatal(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
