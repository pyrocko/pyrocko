import unittest
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

from pyrocko.snuffler import Snuffler, SnufflerWindow
from pyrocko.pile import Pile
from pyrocko import gui_util


class GUITest(unittest.TestCase):

    def test_add_remove_marker(self):
        self.snuffler = Snuffler()
        p = Pile()

        win = SnufflerWindow(pile=p, show=False)
        pile_viewer = win.pile_viewer
        viewer = pile_viewer.viewer

        # Pick
        QTest.mouseDClick(viewer, Qt.LeftButton)
        QTest.mouseRelease(viewer, Qt.LeftButton)
        QTest.mouseClick(viewer, Qt.LeftButton)

        self.assertEqual(viewer.get_active_event(), None)

        # select all visible markers
        QTest.keyPress(pile_viewer, 'a')
        self.assertEqual(len(viewer.get_markers()), 1)

        # convert to EventMarker
        QTest.keyPress(pile_viewer, 'e')
        self.assertTrue(
            isinstance(viewer.get_markers()[0], gui_util.EventMarker))

        # remove selected markers
        QTest.keyPress(pile_viewer, Qt.Key_Backspace)
        self.assertEqual(len(viewer.get_markers()), 0)

        # quit
        QTest.keyPress(pile_viewer, 'q')

if __name__ == '__main__':
    unittest.main()
