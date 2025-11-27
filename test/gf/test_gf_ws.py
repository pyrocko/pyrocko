import unittest
import shutil
import os
import threading
import logging
import tempfile
import time
import asyncio

from pyrocko.gf import LocalEngine, store, ws
from pyrocko import util
from pyrocko.fomosto import ahfullgreen

op = os.path
logger = logging.getLogger('pyrocko.test.test_gf_ws')


class GFWSTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()

        cls.serve_dir = tempfile.mkdtemp(prefix='pyrocko')

        ahfullgreen.init(cls.serve_dir, None)
        cls.store = store.Store(cls.serve_dir, 'r')
        cls.store.make_travel_time_tables()
        cls.store_id = cls.store.config.id
        ahfullgreen.build(cls.serve_dir)

        cls.dl_dir = tempfile.mkdtemp(prefix='pyrocko')
        os.chdir(cls.dl_dir)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        shutil.rmtree(cls.serve_dir)
        shutil.rmtree(cls.dl_dir)

    def test_local_server(self):
        from pyrocko.gf import server

        loop = asyncio.new_event_loop()

        class ServerThread(threading.Thread):
            def __init__(self, serve_dir):
                threading.Thread.__init__(self)
                self.engine = LocalEngine(
                    store_dirs=[serve_dir])

            def run(self):
                asyncio.set_event_loop(loop)
                server.run(self.engine, host='localhost', port=32483)

        t_ws = ServerThread(self.serve_dir)
        t_ws.start()

        try:
            time.sleep(1)
            ws.download_gf_store(
                site='http://localhost:32483',
                store_id=self.store_id,
                quiet=False)
            gfstore = store.Store(self.store_id)
            gfstore.check()

        finally:
            loop.call_soon_threadsafe(server.stop)
            t_ws.join(5.)


if __name__ == '__main__':
    util.setup_logging('test_gf_ws', 'warning')
    unittest.main()
