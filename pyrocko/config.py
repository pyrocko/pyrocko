import os

earthradius = 6371.*1000.
#cache_dir = '/tmp/pyrocko_0.3_cache_%s' % os.environ['USER']
cache_dir = os.path.join(os.environ['HOME'], '.pyrocko_0.3_cache')
