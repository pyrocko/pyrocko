from __future__ import division, print_function, absolute_import
import unittest
from .. import common

import pyrocko.trace
from pyrocko import util, io, model, pile

if common.have_obspy():
    import obspy
    from pyrocko import obspy_compat
    obspy_compat.plant()


def close_win(win):
    win.close()


@common.require_obspy
class ObsPyCompatTestCase(unittest.TestCase):

    @common.require_gui
    def test_obspy_snuffle(self):
        fn = common.test_data_file('test1.mseed')

        stream = obspy.read(fn)
        stream.snuffle(launch_hook=close_win)

        trace = stream[0]
        trace.snuffle(launch_hook=close_win)

    @common.require_gui
    def test_obspy_fiddle(self):
        fn = common.test_data_file('test1.mseed')

        stream = obspy.read(fn)
        stream2 = stream.fiddle(launch_hook=close_win)  # noqa

        trace = stream[0]
        trace2 = trace.fiddle(launch_hook=close_win)  # noqa

    def test_to_obspy_trace(self):
        traces = io.load(common.test_data_file('test1.mseed'))
        for tr in traces:
            obs_tr = tr.to_obspy_trace()

            assert isinstance(obs_tr, obspy.Trace)
            assert obs_tr.data.size == tr.data_len()

            obs_stats = obs_tr.stats
            for attr in ('network', 'station', 'location', 'channel'):
                assert obs_stats.__getattr__(attr) == tr.__getattribute__(attr)

    def test_to_obspy_stream(self):
        pl = pile.Pile()
        pl.load_files([common.test_data_file('test1.mseed')],
                      show_progress=False)
        st = pl.to_obspy_stream()

        assert isinstance(st, obspy.Stream)
        assert len(st) == len([tr for tr in pl.iter_all()])
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
        from obspy.clients.fdsn.client import Client
        client = Client('IRIS')
        cat = client.get_events(eventid=609301)
        events = cat.to_pyrocko_events()
        self.assertEqual(len(events), len(cat))


if __name__ == "__main__":
    util.setup_logging('test_obspy_compat', 'warning')
    unittest.main()
