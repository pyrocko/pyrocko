# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from pyrocko.squirrel import error, environment, database, base
from .. import common


def setup(subparsers):
    p = common.add_parser(
        subparsers, 'persistent',
        help='Manage persistent selections.',
        description='''Manage persistent selections.

Usually, the contents of files given to Squirrel are made available within the
application through a runtime selection which is discarded again when the
application quits. Getting the cached meta-data into the runtime selection can
be a bottleneck for application startup with large datasets. To speed up
startup of Squirrel-based applications, persistent selections created with the
--persistent option can be used.

This command allows to list and delete persistent selections.
''')

    p.add_argument(
        'action',
        choices=['list', 'delete'],
        help='Select action to perform.')

    p.add_argument(
        'names',
        nargs='*',
        help='Persistent selection names.')

    return p


def call(parser, args):
    env = environment.get_environment()
    db = database.get_database(env.expand_path(env.database_path))

    available = sorted(db.get_persistent_names())
    for name in args.names:
        if name not in available:
            raise error.SquirrelError(
                'No such persistent selection: %s' % name)

    if args.action == 'list':
        if not args.names:
            names = available
        else:
            names = args.names

        for name in names:
            print(name)

    elif args.action == 'delete':
        for name in args.names:
            sq = base.Squirrel(persistent=name)
            sq.delete()

    else:
        raise error.SquirrelError('Invalid action: %s' % args.action)
