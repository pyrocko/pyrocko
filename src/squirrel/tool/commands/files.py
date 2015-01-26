# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from .. import common


def setup(subparsers):
    p = common.add_parser(
        subparsers, 'files',
        help='Lookup files providing given content selection.')

    common.add_selection_arguments(p)
    common.add_query_arguments(p)
    return p


def call(parser, args):
    d = common.squirrel_query_from_arguments(args)
    squirrel = common.squirrel_from_selection_arguments(args)

    paths = set()
    if d:
        for nut in squirrel.iter_nuts(**d):
            paths.add(nut.file_path)

        for path in sorted(paths):
            print(path)
    else:
        for path in squirrel.iter_paths():
            print(path)
