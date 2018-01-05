from __future__ import division, print_function, absolute_import
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
logger = logging.getLogger('pyrocko.test.test_gf_ws')


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
                self.s = server.Server(
                    'localhost', 32483, server.SeismosizerHandler, self.engine)

            def run(self):
                asyncore.loop(1., use_poll=True)

        t_ws = ServerThread(self.serve_dir)
        t_ws.start()

        try:
            ws.download_gf_store(
                site='http://localhost:32483',
                store_id=self.store_id,
                quiet=False)
            gfstore = store.Store(self.store_id)
            gfstore.check()

        finally:
            # import time
            # time.sleep(100)
            t_ws.s.close()
            t_ws.join(1.)


if __name__ == '__main__':
    util.setup_logging('test_gf_ws', 'warning')
    unittest.main()
