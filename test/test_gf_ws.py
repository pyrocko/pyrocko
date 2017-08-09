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

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()

        cls.serve_dir = tempfile.mkdtemp(prefix='pyrocko')

        ahfullgreen.init(cls.serve_dir, None)
        cls.store = store.Store(cls.serve_dir, 'r')
        cls.store.make_ttt()
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

        class ServerThread(threading.Thread):
            def __init__(self, serve_dir):
                threading.Thread.__init__(self)
                self.engine = LocalEngine(
                    store_dirs=[serve_dir])

            def run(self):
                self.s = server.Server(
                    '', 8080, server.SeismosizerHandler, self.engine)
                asyncore.loop(timeout=.2)

        t_ws = ServerThread(self.serve_dir)
        t_ws.start()

        try:
            ws.download_gf_store(
                site='localhost',
                store_id=self.store_id,
                quiet=False)

            gfstore = store.Store(self.store_id)
            gfstore.check()

        finally:
            t_ws.s.close()
            t_ws.join(1.)


if __name__ == '__main__':
    util.setup_logging('test_gf_ws', 'warning')
    unittest.main()
