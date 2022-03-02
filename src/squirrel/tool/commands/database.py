# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import logging

from pyrocko import squirrel as sq

logger = logging.getLogger('psq.cli.database')

description = '''
Actions:

    env           Show current Squirrel environment.
    stats         Show brief information about cached meta-data.
    stats-full    Show detailed information about cached meta-data.
    files         Show paths of files for which cached meta-data is available.
    files-nnuts   Show number of index entries for each known file.
    nuts          Dump index entry summaries.
    cleanup       Remove leftover volatile data entries.
    remove paths  Remove cached meta-data of files matching given patterns.
'''.strip()


def make_subparser(subparsers):
    return subparsers.add_parser(
        'database',
        help='Database inspection and maintenance.',
        description=description)


def setup(parser):
    parser.add_argument('action', nargs='?', default='env', choices=[
        'env', 'stats', 'stats-full', 'files', 'files-nnuts', 'nuts',
        'cleanup', 'remove'])

    parser.add_argument('path_patterns', nargs='*')


def run(parser, args):
    action = args.action
    if action == 'env':
        env = sq.get_environment()
        env.path_prefix = env.get_basepath()
        print(env)

    else:
        s = sq.Squirrel()
        db = s.get_database()
        if action == 'stats':
            print(db.get_stats())
        elif action == 'stats-full':
            print(db.get_stats().dump())

        elif action == 'files':
            for path in db.iter_paths():
                print(path)

        elif action == 'files-nnuts':
            for path, nnuts in db.iter_nnuts_by_file():
                print(path, nnuts)

        elif action == 'nuts':
            for path, nuts in db.undig_all():
                print(path)
                for nut in nuts:
                    print('  ' + nut.summary)

        elif action == 'cleanup':
            n_removed = db._remove_volatile()
            logger.info('Number of entries removed: %i' % n_removed)

        elif action == 'remove':
            if not args.path_patterns:
                raise sq.SquirrelError(
                    'No path patterns to remove have been specified.')

            n_removed = 0
            for path in args.path_patterns:
                n_removed += db.remove_glob(path)

            logger.info('Number of entries removed: %i' % n_removed)
