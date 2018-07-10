from __future__ import division, print_function, absolute_import
import os
from . import common


if not os.environ.get('MPL_SHOW', False):
    common.matplotlib_use_agg()
