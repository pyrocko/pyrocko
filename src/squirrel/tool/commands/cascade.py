# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`squirrel cascade`.
'''

import logging

from pyrocko import progress
from ..common import ldq
from pyrocko.squirrel.error import ToolError


logger = logging.getLogger('psq.cli.cascade')

headline = 'Create hierarchical data overviews.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'cascade',
        help=headline,
        description=headline)


method_choices = ('mean', 'min', 'max')


def setup(parser):
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments()

    parser.add_argument(
        '--levels',
        dest='nlevels',
        type=int,
        metavar='INT',
        default=10,
        help='Number of levels to create.')

    parser.add_argument(
        '--fold',
        dest='nfold',
        type=int,
        metavar='INT',
        default=2,
        help='Decimation factor applied in each level.')

    parser.add_argument(
        '--methods',
        dest='methods',
        default='mean,min,max',
        help='Methods of reduction, comma-separated list. '
             'Choices: %s Default: ``%s``' % (
                 ldq(method_choices), 'mean,min,max'))

    parser.add_argument(
        '--out-storage-path',
        dest='out_storage_path',
        metavar='PATH',
        help='Store output in directory PATH.')

    parser.add_argument(
        '--force',
        dest='force',
        action='store_true',
        default=False,
        help='Force overwriting of existing files.')

    parser.add_argument(
        '--append',
        dest='append',
        action='store_true',
        default=False,
        help='Append to existing files. Checks are preformed to ensure that '
             'appended data has no overlap with already existing data.')

    parser.add_argument(
        '--merge',
        dest='merge',
        action='store_true',
        default=False,
        help='Merge with existing data in files.')


def run(parser, args):
    from pyrocko.squirrel.cascade import cascade
    methods = set()
    for method in [s.strip() for s in args.methods.split(',')]:
        if method not in method_choices:
            raise ToolError(
                'Available reduction methods are: %s'
                % ', '.join(method_choices))

        methods.add(method)

    methods = sorted(list(methods))

    if not args.out_storage_path:
        raise ToolError(
            'Specify output storage directory with --out-storage-path')

    squirrel = args.make_squirrel()
    with progress.view():
        cascade(
            squirrel,
            nlevels=args.nlevels,
            storage_path=args.out_storage_path,
            nfold=args.nfold,
            methods=methods,
            force=args.force,
            append=args.append,
            merge=args.merge,
            **args.squirrel_query)
