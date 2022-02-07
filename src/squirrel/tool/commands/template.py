# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function

import re

from pyrocko.guts import load_string, dump

from .. import common

path_prefix = '''
# All file paths given below are treated relative to the location of this
# configuration file. Here we may give a common prefix. For example, if the
# configuration file is in the sub-directory 'PROJECT/config/', set it to '..'
# so that all paths are relative to 'PROJECT/'.
path_prefix: '.'
'''.strip()


def _template_online_dataset(**kwargs):
    lqargs = []
    for k in ['network', 'station', 'channel']:
        if k in kwargs:
            v = kwargs.pop(k)
            lqargs.append("    %s: '%s'" % (k, v))

    kwargs['qargs'] = '\n' + '\n'.join(lqargs) if lqargs else '{}'
    return '''
--- !squirrel.Dataset

{path_prefix}

# Data sources to be added (LocalData, FDSNSource, CatalogSource, ...)
sources:
- !squirrel.FDSNSource

  # URL or alias of FDSN site.
  site: {site}

  # FDSN query arguments to make metadata queries.
  # See http://www.fdsn.org/webservices/fdsnws-station-1.1.pdf
  # Time span arguments should not be added here, because they are handled
  # automatically by Squirrel.
  query_args: {qargs}
'''.format(path_prefix=path_prefix, **kwargs).strip()


templates = {
    'dataset-local': {
        'description':
            'A typical collection of local files.',
        'yaml': '''
--- !squirrel.Dataset

{path_prefix}

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
'''.format(path_prefix=path_prefix).strip()},

    'dataset-geofon': {
        'description':
            'Everything available through GEOFON.',
        'yaml': _template_online_dataset(site='geofon')
    },

    'dataset-iris-seis': {
        'description':
            'All high- and low-gain seismometer channels at IRIS.',
        'yaml': _template_online_dataset(
                site='iris', channel='?H?,?L?')
    },

    'dataset-iris-seis-bb': {
        'description':
            'All broad-band high-gain seismometer channels at IRIS.',
        'yaml': _template_online_dataset(
                site='iris', channel='VH?,LH?,BH?,HH?')
    },

    'dataset-bgr-gr-bfo': {
        'description': 'Station GR.BFO from BGR.',
        'yaml': _template_online_dataset(
                site='iris', network='GR', station='BFO')
    },
}

names = sorted(templates.keys())

template_listing = '\n'.join(
    '%-30s %s' % (
        '%s:' % name,
        templates[name]['description']) for name in templates)


def setup(subparsers):
    p = common.add_parser(
        subparsers, 'template',
        help='Print configuration snippets.',
        description='''
Available configuration SNIPPETs:

{}
'''.format(template_listing).strip())

    p.add_argument(
        'name',
        choices=names,
        nargs='?',
        metavar='SNIPPET',
        help='Name of template snippet to print.')

    p.add_argument(
        '--format', '-f',
        choices=['commented', 'normal', 'brief'],
        default='commented',
        metavar='FMT',
        help='Set verbosity level of output YAML. Choices: %(choices)s. '
             'Default: %(default)s.')

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

    if not args.name:
        print(template_listing)

    else:
        func = {
            'brief': brief,
            'commented': lambda s: s,
            'normal': decomment}[args.format]

        print(func(templates[args.name]['yaml']))
