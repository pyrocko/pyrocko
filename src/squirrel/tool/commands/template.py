# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import re

from pyrocko.guts import load_string, dump

from .. import common


templates = {
    'dataset-local': '''
--- !squirrel.Dataset

# All file paths referenced below are treated relative to the location of this
# configuration file, here we may give a common prefix. E.g. setting it to '..'
# if the configuration file is in the sub-directory '${project_root}/config'
# allows us to give the paths below relative to '${project_root}'.
path_prefix: '.'

# Data sources to be added (LocalData, FDSNSource, CatalogSource, ...)
sources:
- !squirrel.LocalData  # This data source is for local files.

  # These paths are scanned for waveforms, stations, events.
  paths:
  - 'catalogs/events.txt'
  - 'meta/stations.xml'
  - 'data/waveforms'

  # Select file format or 'detect' for autodetection.
  format: 'detect'
'''}

names = sorted(templates.keys())


def setup(subparsers):
    p = common.add_parser(
        subparsers, 'template',
        help='Print config snippets.')

    p.add_argument(
        'name',
        choices=names,
        help='Name of template to print (choices: %(choices)s).')

    p.add_argument(
        '--format', '-f',
        choices=['commented', 'normal', 'brief'],
        default='normal',
        help='Set verbosity level of output YAML (default: %(default)s).')

    return p


def decomment(s):
    out = []
    for line in s.splitlines():
        line = re.sub(r'#.+', '', line)
        if line.strip():
            out.append(line)

    return '\n'.join(out)


def brief(s):
    return dump(load_string(s))


def call(parser, args):

    func = {
        'brief': brief,
        'commented': lambda s: s,
        'normal': decomment}[args.format]

    print(func(templates[args.name]))
