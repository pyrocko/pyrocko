# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

headline = 'Search indexed contents.'


def make_subparser(subparsers):
    return subparsers.add_parser(
        'nuts',
        help=headline,
        description=headline)


def setup(parser):
    parser.add_squirrel_selection_arguments()
    parser.add_squirrel_query_arguments()

    parser.add_argument(
        '--contents',
        action='store_true',
        dest='print_contents',
        default=False,
        help='Print contents.')


def run(parser, args):
    squirrel = args.make_squirrel()
    for nut in squirrel.iter_nuts(**args.squirrel_query):
        if args.print_contents:
            print('# %s' % nut.summary)
            print(squirrel.get_content(nut).dump())
        else:
            print(nut.summary)
