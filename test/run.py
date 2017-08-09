import sys
import argparse
import unittest
import pyrocko.util
from os import path
import matplotlib
matplotlib.use('Agg')


def get_test_suite(loader=None, pattern='test_*.py'):
    if loader is None:
        loader = unittest.defaultTestLoader

    return loader.discover(path.dirname(__file__), pattern)


def load_tests(loader, tests, pattern):
    return get_test_suite(loader)


all = get_test_suite()

gf = get_test_suite(pattern='test_gf_*.py')

trace = get_test_suite(pattern='test_trace.py')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        add_help=False,
        description='Pyrocko test runner '
                    '(for unittest standard options, see below)')

    parser.add_argument(
        '--coverage', action='store_true',
        help='Measure coverage of tests')
    parser.add_argument(
        '-h', '--help', action='store_true',
        help='Show help messages and exit')
    options, args = parser.parse_known_args()

    argv = [sys.argv[0]] + args
    if options.help:
        parser.print_help()
        argv.append('-h')
        print()

    pyrocko.util.setup_logging('test_all', 'warning')

    if options.coverage and not options.help:
        import coverage
        cov = coverage.Coverage(source=['pyrocko'])
        cov.start()

    unittest.main(argv=argv, exit=False)

    if options.coverage and not options.help:
        cov.stop()
        cov.save()
        cov.report()
        cov.html_report()
