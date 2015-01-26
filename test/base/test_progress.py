from __future__ import print_function, absolute_import

import time
import unittest
import logging

from pyrocko import progress, util

logger = logging.getLogger('test_progress')


class ProgressTestCase(unittest.TestCase):

    def demo_terminal_status_window(self):
        self.test_terminal_status_window(slow=True)

    def test_terminal_status_window(self, slow=False):
        util.setup_logging('test_progress', 'info')

        frames = []
        for iframe in range(11):
            lines = []
            for iline in range(5-abs(iframe-5)):
                lines.append(str(iframe) * (iframe % 4))

            frames.append(lines)

        with progress.TerminalStatusWindow() as t:
            for iframe, lines in enumerate(frames):
                logger.info('frame %i' % iframe)
                t.draw(lines)
                if slow:
                    time.sleep(1.0)
                else:
                    time.sleep(0.1)

    def demo_progress(self):
        util.setup_logging('test_progress', 'info')
        self.test_progress(slow=True)

    def test_progress(self, slow=False):

        p = progress.Progress()

        t1 = p.task('a task', logger=logger)
        t2 = p.task('another task', 500, logger=logger)

        with p.view():

            s = [0, 0, 0]
            for i in range(1000):
                if i > 300:
                    s[0] += 1
                    t1.update(s[0])

                s[1] += 2

                if i == 500:
                    t2.done()

                if i == 200:
                    t3 = p.task('late task', 20, logger=logger)

                if i > 200 and i % 5 == 0:
                    s[2] += 1
                    t3.update(s[2])

                t2.update(s[1])
                if s[2] == 20:
                    t3.done()

                if slow:
                    time.sleep(.01)
                else:
                    time.sleep(.0005)

                if s[0] == 666:
                    t1.fail('oh damn...')


if __name__ == "__main__":
    unittest.main()
