import unittest
from pyrocko.model import quakeml
from pyrocko import util
import common


class QuakeMLTestCase(unittest.TestCase):

    def test_read(self):

        fpath = common.test_data_file('test.quakeml')
        qml = quakeml.QuakeML.load_xml(filename=fpath)
        events = qml.get_pyrocko_events()
        assert len(events) == 1
        e = events[0]
        assert e.lon == -116.9945
        assert e.lat == 33.986
        assert e.depth == 17300
        assert e.time == util.stt("1999-04-02 17:05:10.500")


if __name__ == "__main__":
    util.setup_logging('test_quakeml', 'warning')
    unittest.main()
