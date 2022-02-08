# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from .. import common


def setup_subcommand(subparsers):
    return common.add_parser(
        subparsers, 'files',
        help='Lookup files providing given content selection.')


def setup(parser):
    parser.add_argument(
        '--relative',
        action='store_true',
        default=False,
        help='Reveal path as it is stored in the database. This is relative '
             'for files inside a Squirrel environment.')

    common.add_selection_arguments(parser)
    common.add_query_arguments(parser)


def call(parser, args):
    d = common.squirrel_query_from_arguments(args)
    squirrel = common.squirrel_from_selection_arguments(args)

    paths = set()
    if d:
        for nut in squirrel.iter_nuts(**d):
            paths.add(nut.file_path)

        db = squirrel.get_database()
        for path in sorted(paths):
            print(db.relpath(path) if args.relative else path)

    else:
        for path in squirrel.iter_paths(raw=args.relative):
            print(path)
