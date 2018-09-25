# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import os
import re

op = os.path


modules = [m for m in os.listdir(op.dirname(__file__))
           if re.search(r'[^_(dummy)].py', m)]
AVAILABLE_BACKENDS = set([op.splitext(m)[0] for m in modules])
