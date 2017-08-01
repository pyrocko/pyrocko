# python 2/3
import matplotlib
matplotlib.use('PS')

import unittest # noqa
import logging # noqa
import os # noqa
from pyrocko import util # noqa

logger = logging.getLogger('test_tutorials')


to_test = {
    'automap_example': ['stations_deadsea.pf'],
    'markers_example1': ['my_markers.pf'],
    'trace_handling_example_pz':
    ['displacement.mseed', 'STS2-Generic.polezero.txt', 'test.mseed'],
    'beachball_example01': ['beachball-example03.pdf'],
    'beachball_example02': ['beachball-example02.pdf'],
    'beachball_example03': ['beachball-example03.pdf'],
    'beachball_example04': ['beachball-example04.png'],
    'gf_forward_example1': None,
    'gf_forward_example2': None,
    'gf_forward_example3': None,
    'gf_forward_example4': None,
    'gshhg_example': None,
    'tectonics_example': None,
    'test_response_plot': None,
}


def cleanup(fns):
    if fns:
        for fn in fns:
            try:
                os.remove(fn)
            except OSError:
                pass


class TutorialTestCase(unittest.TestCase):

    def test_all(self):
        for m, to_be_removed in to_test.items():
            try:
                module = __import__('pyrocko.tutorials.' + m)
                module()
            except Exception as e:
                logger.exception('%s - %s' % (m, e))
            finally:
                cleanup(to_be_removed)


if __name__ == '__main__':
    util.setup_logging('test_tutorials', 'warning')
    unittest.main()
