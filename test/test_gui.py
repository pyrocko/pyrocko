import unittest
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

from pyrocko.snuffler import Snuffler, SnufflerWindow
from pyrocko.pile import make_pile
from pyrocko import pile_viewer as pyrocko_pile_viewer
from pyrocko import gui_util, util
from pyrocko import config


class GUITest(unittest.TestCase):

    def setUp(self):
        self.snuffler = Snuffler()
        p = make_pile('data/test2.mseed')
        win = SnufflerWindow(pile=p, show=False)
        self.pile_viewer = win.pile_viewer
        self.viewer = self.pile_viewer.viewer

    def tearDown(self):
        QTest.keyPress(self.pile_viewer, 'q')

    def test_markers(self):
        QTest.mouseDClick(self.viewer, Qt.LeftButton)

        # This should be done by mouseDClick, actually....
        QTest.mouseRelease(self.viewer, Qt.LeftButton)
        QTest.mouseClick(self.viewer, Qt.LeftButton)

        self.assertEqual(self.viewer.get_active_event(), None)

        conf = config.config('snuffler')

        # test kinds and phases
        kinds = range(5)
        fkey_map = pyrocko_pile_viewer.fkey_map

        for k in kinds:
            for fkey, fkey_int in fkey_map.items():
                fkey_int += 1
                QTest.keyPress(self.pile_viewer, fkey)
                QTest.keyPress(self.pile_viewer, str(k))

                if fkey_int != 10:
                    want = conf.phase_key_mapping.get(
                        "F%s" % fkey_int, 'Undefined')
                else:
                    want = None
                m = self.viewer.get_markers()[0]
                self.assertEqual(m.kind, k)
                if want:
                    self.assertEqual(m.get_phasename(), want)

        # write markers:
        QTest.mouseClick(self.viewer.menu, Qt.LeftButton)

        # cleanup
        QTest.keyPress(self.pile_viewer, 'a')
        QTest.keyPress(self.pile_viewer, Qt.Key_Backspace)
        self.assertEqual(len(self.viewer.get_markers()), 0)

        QTest.mouseDClick(self.viewer, Qt.LeftButton)
        QTest.mouseRelease(self.viewer, Qt.LeftButton)
        QTest.mouseClick(self.viewer, Qt.LeftButton)
        # select all visible markers
        QTest.keyPress(self.pile_viewer, 'a')
        self.assertEqual(len(self.viewer.get_markers()), 1)

        # convert to EventMarker
        QTest.keyPress(self.pile_viewer, 'e')
        self.assertTrue(
            isinstance(self.viewer.get_markers()[0], gui_util.EventMarker))

        # cleanup
        QTest.keyPress(self.pile_viewer, Qt.Key_Backspace)
        self.assertEqual(len(self.viewer.get_markers()), 0)

if __name__ == '__main__':
    util.setup_logging('test_gui', 'warning')
    unittest.main()
