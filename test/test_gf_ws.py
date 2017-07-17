import unittest
import shutil
import os
import asyncore
import threading
import logging

from pyrocko.gf import server, LocalEngine, ws, store
from pyrocko import util


logger = logging.getLogger('test_gf_ws.py')


class GFWSTestCase(unittest.TestCase):

    def test_local_server(self):
        class ServerThread(threading.Thread):
            def __init__(self):
                threading.Thread.__init__(self)
                self.engine = LocalEngine(store_superdirs=['data'])

            def run(self):
                self.s = server.Server(
                    '', 8080, server.SeismosizerHandler, self.engine)
                asyncore.loop(timeout=.2)

        store_id = 'test_store'

        t_ws = ServerThread()
        t_ws.start()

        if os.path.exists(store_id):
            shutil.rmtree(store_id)

        ws.download_gf_store(site='localhost', store_id=store_id)
        t_ws.s.close()
        t_ws.join()

        gfstore = store.Store(store_id)
        gfstore.check()

        # cleanup
        shutil.rmtree(store_id)


if __name__ == '__main__':
    util.setup_logging('test_gf_ws', 'warning')
    unittest.main()
