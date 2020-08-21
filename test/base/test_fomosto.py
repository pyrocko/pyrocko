# python 2/3
from __future__ import division, print_function, absolute_import

import unittest

from .. import common
from pyrocko import util

util.force_dummy_progressbar = True

refs = '''
@software{pyrocko,
  title = {Pyrocko: {{A}} Versatile Seismology Toolkit for {{Python}}.},
  rights = {GPLv3},
  url = {http://pyrocko.org},
  shorttitle = {Pyrocko},
  year = 2017,
  version = {2018.1.29},
  urldate = {2018-02-23},
  keywords = {Scientific/Engineering,Scientific/Engineering},
  author = {The Pyrocko Developers},
  doi = {10.5880/GFZ.2.1.2017.001}
}
'''


def fomosto(*args, **kwargs):
    # kwargs['tee'] = True
    common.call('fomosto', *args, **kwargs)


def dummy(*args, **kwargs):
    pass


class FomostoTestCase(unittest.TestCase):

    def test_fomosto_usage(self):
        common.call_assert_usage('fomosto')

    def test_fomosto_ahfull(self):
        with common.run_in_temp():
            fomosto('init', 'ahfullgreen', 'my_gfs')
            with common.chdir('my_gfs'):
                common.call_assert_usage('fomosto', '--help')
                common.call_assert_usage('fomosto', 'help')
                common.call_assert_usage('fomosto', 'init')
                fomosto('ttt')
                fomosto('build', '--nworkers=2')
                fomosto('stats')
                fomosto('check')
                fomosto('tttview', 'anyP')
                fomosto('extract', '@2,@2')
                fomosto('tttextract', 'anyP', '@2,@2')
                fomosto('tttextract', 'anyP', '@2,@2', '--output=anyP.phase')
                fomosto('modelview')
                fomosto('qc')
                fomosto('decimate', '2')
                fomosto('decimate', '4')
                fomosto('upgrade')

            fomosto('init', 'redeploy', 'my_gfs', 'my_gfs2')
            fomosto('redeploy', 'my_gfs', 'my_gfs2')
            fomosto('ttt', 'my_gfs2')

            from pyrocko import trace
            snuffle, trace.snuffle = trace.snuffle, dummy
            fomosto('view', '--show-phases=all', '--extract=@2,@2',
                    'my_gfs', 'my_gfs2')
            trace.snuffle = snuffle

            fomosto('modelview', '--parameters=vp/vs,rho', 'my_gfs', 'my_gfs2')

    def test_fomosto_ahfull_refs(self):
        with common.run_in_temp():
            fomosto('init', 'ahfullgreen', 'my_gfs3')
            with common.chdir('my_gfs3'):
                try:
                    from pybtex.database.input import bibtex  # noqa
                    with open('refs.bib', 'w') as f:
                        f.write(refs)
                    fomosto('addref', 'refs.bib')
                except ImportError as e:
                    raise unittest.SkipTest(str(e))
