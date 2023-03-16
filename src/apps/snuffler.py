# https://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from pyrocko.gui.snuffler import snuffler


def main(args=None):
    snuffler.snuffler_from_commandline(args=args)


if __name__ == '__main__':
    main()
