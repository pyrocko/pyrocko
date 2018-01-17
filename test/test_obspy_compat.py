from __future__ import division, print_function, absolute_import
import unittest
from . import common

import pyrocko.trace
from pyrocko import util, io, model, pile

import obspy
from pyrocko import obspy_compat
obspy_compat.plant()


class ObsPyCompatTestCase(unittest.TestCase):

    @unittest.skipUnless(
        common.have_gui(),
        'No GUI available')
    def test_obspy_snuffle(self):
        fn = common.test_data_file('test1.mseed')

        stream = obspy.read(fn)
        stream.snuffle()

        trace = stream[0]
        trace.snuffle()

    def test_to_obspy_trace(self):
        traces = io.load(common.test_data_file('test1.mseed'))
        for tr in traces:
            assert isinstance(tr.to_obspy_trace(), obspy.Trace)

    def test_to_obspy_stream(self):
        pl = pile.Pile()
        pl.load_files([common.test_data_file('test1.mseed')],
                      show_progress=False)
        st = pl.to_obspy_stream()

        assert isinstance(st, obspy.Stream)
        for tr in st:
            assert isinstance(tr, obspy.Trace)

    def test_to_pyrocko_traces(self):
        st = obspy.read(common.test_data_file('test1.mseed'))

        traces = st.to_pyrocko_traces()
        assert isinstance(traces, list)
        for tr in traces:
            assert isinstance(tr, pyrocko.trace.Trace)

        for tr in st:
            assert isinstance(tr.to_pyrocko_trace(), pyrocko.trace.Trace)

    def test_to_pyrocko_stations(self):
        fn = common.test_data_file('geeil.geofon.xml')
        inventory = obspy.read_inventory(fn)

        for sta in inventory.to_pyrocko_stations():
            assert isinstance(sta, model.Station)

    def test_to_pyrocko_events(self):
        pass


if __name__ == "__main__":
    util.setup_logging('test_obspy_compat', 'warning')
    unittest.main()
