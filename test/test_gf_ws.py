import unittest
import shutil
import os
import asyncore
import threading
import logging
import tempfile

from pyrocko.gf import server, LocalEngine, ws, store
from pyrocko import util
from pyrocko.fomosto import ahfullgreen

op = os.path
logger = logging.getLogger('test_gf_ws.py')


class GFWSTestCase(unittest.TestCase):

    def setUp(self):
        self.store_dir = tempfile.mkdtemp(prefix='pyrocko')
        ahfullgreen.init(self.store_dir, None)

        self.store = store.Store(self.store_dir, 'r')
        self.store.make_ttt()
        self.store_id = self.store.config.id

        ahfullgreen.build(self.store_dir)

    def tearDown(self):
        shutil.rmtree(self.store_dir)

    def test_local_server(self):

        class ServerThread(threading.Thread):
            def __init__(self, store_dir):
                threading.Thread.__init__(self)
                self.engine = LocalEngine(
                    store_dirs=[store_dir])

            def run(self):
                self.s = server.Server(
                    '', 8080, server.SeismosizerHandler, self.engine)
                asyncore.loop(timeout=.2)

        t_ws = ServerThread(self.store_dir)
        t_ws.start()

        try:
            ws.download_gf_store(
                site='localhost',
                store_id=self.store_id,
                quiet=True)

            gfstore = store.Store(self.store_id)
            gfstore.check()

        finally:
            shutil.rmtree(self.store_id)
            t_ws.s.close()
            t_ws.join()


if __name__ == '__main__':
    util.setup_logging('test_gf_ws', 'warning')
    unittest.main()
