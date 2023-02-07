# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

import pkgutil

command_modules = []

for _, modname, ispkg in pkgutil.iter_modules(__path__, __name__ + '.'):
    command_modules.append(__import__(modname, fromlist='dummy'))
