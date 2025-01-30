# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Implementation of :app:`gato mantra`.
'''

from pyrocko import squirrel, gato


headline = 'Print example mantra configurations.'

description = ''


def make_subparser(subparsers):
    return subparsers.add_parser(
        'mantra',
        help=headline,
        description=headline + '\n\n' + description)


def setup(parser):
    pass


def run(parser, args):

    print(
        squirrel.Mantra(
            name='acme',
            operators=[
                squirrel.Restitution(
                    frequency_min=1.0,
                    frequency_max=10.0),
                squirrel.ToENZ(),
                gato.ACMEOperator(
                    codes=['*.*.*.*Z.*'],
                    downsampling_deltat=0.02,
                    whitening_bandwidth=1.0,
                    time_normalization_deltat=5.,
                    time_window=60.,
                    nsubwindows=10)]))
