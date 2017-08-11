from __future__ import division, print_function, absolute_import
import unittest
try:
    from . import common
except:
    import common
import numpy as num
import tempfile
import os

if common.have_gui():  # noqa
    from PyQt4.QtTest import QTest
    from PyQt4.QtCore import Qt, QPoint
    from PyQt4.QtGui import QStyleOptionSlider, QStyle
    from pyrocko.gui.snuffler import Snuffler, SnufflerWindow
    from pyrocko.gui import pile_viewer as pyrocko_pile_viewer
    from pyrocko.gui import gui_util
    from pyrocko import util, model


from pyrocko.pile import make_pile
from pyrocko import config, trace


@common.require_gui
class GUITest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Create a reusable snuffler instance for all tests cases.
        '''
        super(GUITest, cls).setUpClass()
        cls.snuffler = Snuffler()  # noqa
        fpath = common.test_data_file('test2.mseed')
        p = make_pile(fpath, show_progress=False)
        cls.win = SnufflerWindow(pile=p)
        cls.pile_viewer = cls.win.pile_viewer
        pv = cls.pile_viewer
        cls.main_control_defaults = dict(
            highpass_control=pv.highpass_control.get_value(),
            lowpass_control=pv.lowpass_control.get_value(),
            gain_control=pv.gain_control.get_value(),
            rot_control=pv.rot_control.get_value())

    @classmethod
    def tearDownClass(cls):
        '''
        Quit snuffler.
        '''
        QTest.keyPress(cls.pile_viewer, 'q')

    def setUp(self):
        '''
        reset GUI to initial state
        '''
        for k, v in self.main_control_defaults.items():
            getattr(self.pile_viewer, k).set_value(v)

    def tearDown(self):
        self.clear_all_markers()

    # def test_inputline(self):
    #     pv = self.pile_viewer
    #     il = pv.inputline
    #     QTest.keyPress(pv, ':')
    #     il.setText('ASDF')
    #     QTest.keyPress(pv, )

    def clear_all_markers(self):
        pv = self.pile_viewer
        QTest.keyPress(pv, 'A', Qt.ShiftModifier, 10)
        QTest.keyPress(pv, Qt.Key_Backspace)
        self.assertEqual(len(pv.viewer.get_markers()), 0)

    def click_menu_item(self, qmenu, action_name):
        ''' Emulate a mouseClick on a menu item *action_name* in the
        *qmenu*.'''
        for iaction, action in enumerate(qmenu.actions()):
            if action.text() == action_name:
                QTest.keyClick(qmenu, Qt.Key_Enter)
                for i in range(iaction):
                    QTest.keyClick(qmenu, Qt.Key_Up)
                qmenu.close()
                break
            else:
                QTest.keyClick(qmenu, Qt.Key_Down)

    def get_slider_position(self, slider):
        style = slider.style()
        opt = QStyleOptionSlider()
        return style.subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle)

    def drag_slider(self, slider):
        ''' Click *slider*, drag from one side to the other, release mouse
        button repeat to restore inital state'''
        position = self.get_slider_position(slider)
        QTest.mouseMove(slider, pos=position.topLeft())
        QTest.mousePress(slider, Qt.LeftButton)
        QTest.mouseMove(slider, pos=position.bottomRight())
        QTest.mouseRelease(slider, Qt.LeftButton)
        QTest.mousePress(slider, Qt.LeftButton)
        QTest.mouseMove(slider, pos=position.topLeft())
        QTest.mouseRelease(slider, Qt.LeftButton)

    def test_save_image(self):
        tempfn_svg = tempfile.mkstemp()[1] + '.svg'
        self.pile_viewer.viewer.savesvg(fn=tempfn_svg)
        os.remove(tempfn_svg)

        tempfn_png = tempfile.mkstemp()[1] + '.png'
        self.pile_viewer.viewer.savesvg(fn=tempfn_png)
        os.remove(tempfn_png)

    def add_one_pick(self):
        '''Add a single pick to pile_viewer'''
        pv = self.pile_viewer
        QTest.mouseDClick(pv.viewer, Qt.LeftButton)
        position_tl = pv.pos()
        geom = pv.frameGeometry()
        QTest.mouseMove(pv.viewer, pos=position_tl)
        QTest.mouseMove(pv.viewer, pos=(
            QPoint(position_tl.x()+geom.x()/2., position_tl.y()+geom.y()/2.)))

        # This should be done also by mouseDClick().
        QTest.mouseRelease(pv.viewer, Qt.LeftButton)
        QTest.mouseClick(pv.viewer, Qt.LeftButton)

    def test_main_control_sliders(self):
        self.drag_slider(self.pile_viewer.highpass_control.slider)
        self.drag_slider(self.pile_viewer.lowpass_control.slider)
        self.drag_slider(self.pile_viewer.gain_control.slider)
        self.drag_slider(self.pile_viewer.rot_control.slider)

    def test_markers(self):
        self.add_one_pick()
        pv = self.pile_viewer
        self.assertEqual(pv.viewer.get_active_event(), None)

        conf = config.config('snuffler')

        # test kinds and phases
        kinds = range(5)
        fkey_map = pyrocko_pile_viewer.fkey_map

        for k in kinds:
            for fkey, fkey_int in fkey_map.items():
                fkey_int += 1
                QTest.keyPress(pv, fkey)
                QTest.keyPress(pv, str(k))

                if fkey_int != 10:
                    want = conf.phase_key_mapping.get(
                        "F%s" % fkey_int, 'Undefined')
                else:
                    want = None
                m = pv.viewer.get_markers()[0]
                self.assertEqual(m.kind, k)
                if want:
                    self.assertEqual(m.get_phasename(), want)

    def test_load(self):
        self.pile_viewer.viewer.load('data', regex='\w*.mseed')

    def test_add_traces(self):
        trs = []
        for i in range(3):
            trs.append(
                trace.Trace(network=str(i), tmin=num.random.uniform(1),
                            ydata=num.random.random(100),
                            deltat=num.random.random())
            )
        self.pile_viewer.viewer.add_traces(trs)

    def test_event_marker(self):
        pv = self.pile_viewer
        self.add_one_pick()

        # select all markers
        QTest.keyPress(pv, 'a', Qt.ShiftModifier)

        # convert to EventMarker
        QTest.keyPress(pv, 'e')

        for m in pv.viewer.get_markers():
            self.assertTrue(isinstance(m, gui_util.EventMarker))

        QTest.keyPress(pv, 'd')

    def test_load_save(self):
        nmarkers = 505
        # times = num.random.uniform(-10., 10, nmarkers) # Fails
        times = num.arange(nmarkers)
        markers = [gui_util.Marker(tmin=t, tmax=t,
                                   nslc_ids=[('*', '*', '*', '*'), ])
                   for t in times]

        tempfn = tempfile.mkstemp()[1]

        self.pile_viewer.viewer.add_markers(markers)
        self.pile_viewer.viewer.write_selected_markers(
            fn=tempfn)
        self.pile_viewer.viewer.write_markers(fn=tempfn)
        self.pile_viewer.viewer.read_markers(fn=tempfn)

        for k in 'pnPN':
            QTest.keyPress(self.pile_viewer, k)

        self.pile_viewer.viewer.go_to_time(-20., 20)
        self.pile_viewer.update()
        self.pile_viewer.viewer.update()
        self.pile_viewer.viewer.remove_markers(markers)

        os.remove(tempfn)

    def test_click_non_dialogs(self):
        # Click through many menu option combinations that do not require
        # further interaction. Activate options in pairs of two.

        pv = self.pile_viewer
        tinit = pv.viewer.tmin
        tinitlen = pv.viewer.tmax - pv.viewer.tmin

        non_dialog_actions = [
            'Indivdual Scale',
            'Common Scale',
            'Common Scale per Station',
            'Common Scale per Component',
            'Scaling based on Minimum and Maximum',
            'Scaling based on Mean +- 2 x Std. Deviation',
            'Scaling based on Mean +- 4 x Std. Deviation',
            'Sort by Names',
            'Sort by Distance',
            'Sort by Azimuth',
            'Sort by Distance in 12 Azimuthal Blocks',
            'Sort by Backazimuth',
            '3D distances',
            'Subsort by Network, Station, Location, Channel',
            'Subsort by Network, Station, Channel, Location',
            'Subsort by Station, Network, Channel, Location',
            'Subsort by Location, Network, Station, Channel',
            'Subsort by Channel, Network, Station, Location',
            'Subsort by Network, Station, Channel (Grouped by Location)',
            'Subsort by Station, Network, Channel (Grouped by Location)',
        ]

        options = [
            'Antialiasing',
            'Liberal Fetch Optimization',
            'Clip Traces',
            'Show Boxes',
            'Color Traces',
            'Show Scale Ranges',
            'Show Scale Axes',
            'Show Zero Lines',
            'Fix Scale Ranges',
            'Allow Downsampling',
            'Allow Degapping',
            'FFT Filtering',
            'Bandpass is Lowpass + Highpass',
            'Watch Files'
        ]

        # create an event marker and activate it
        self.add_one_pick()

        keys = 'mAhefrRh+-fgc?'

        def fire_key(x):
            QTest.keyPress(self.pile_viewer, key)

        for key in keys:
            QTest.qWait(100)
            fire_key(key)

        event = model.Event()
        markers = pv.viewer.get_markers()
        self.assertEqual(len(markers), 1)
        markers[0]._event = event

        right_click_menu = self.pile_viewer.viewer.menu
        for action_name in non_dialog_actions:
            for oa in options:
                for ob in options:
                    self.click_menu_item(right_click_menu, action_name)
                    self.click_menu_item(right_click_menu, oa)
                    self.click_menu_item(right_click_menu, ob)

                options.remove(oa)

        self.pile_viewer.viewer.go_to_event_by_name(event.name)
        self.pile_viewer.viewer.go_to_time(tinit, tinitlen)


if __name__ == '__main__':
    util.setup_logging('test_gui', 'warning')
    unittest.main()
