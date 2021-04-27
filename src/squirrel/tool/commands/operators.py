# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

from .. import common
from pyrocko.squirrel.model import separator


guts_prefix = 'squirrel'


def setup(subparsers):
    p = common.add_parser(
        subparsers, 'operators',
        help='Print available operator mappings.')

    common.add_selection_arguments(p)
    return p


def call(parser, args):
    squirrel = common.squirrel_from_selection_arguments(args)

    def scodes(codes):
        css = list(zip(*(c.split(separator) for c in codes)))
        if sum(not all(c == cs[0] for c in cs) for cs in css) == 1:
            return '.'.join(
                cs[0] if all(c == cs[0] for c in cs) else '(%s)' % ','.join(cs)
                for cs in css)
        else:
            return ', '.join(c.replace(separator, '.') for c in codes)

    for operator, in_codes, out_codes in squirrel.get_operator_mappings():
        print('%s <- %s <- %s' % (
            scodes(out_codes), operator.name, scodes(in_codes)))
