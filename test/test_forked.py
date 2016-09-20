import unittest
from pyrocko import forked, util


class ForkedTestCase(unittest.TestCase):

    def testForkedNormal(self):
        try:
            f = forked.Forked(flipped=False)
            f.call('hallo1')
            f.call('hallo2')
            f.close()
        except SystemExit:
            pass

    def testForkedFlipped(self):
        try:
            f = forked.Forked(flipped=True)
            f.call('f hallo1')
            f.call('f hallo2')
            f.close()
        except SystemExit:
            pass

if __name__ == "__main__":
    util.setup_logging('test_forked', 'warning')
    unittest.main()
