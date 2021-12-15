# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

from __future__ import absolute_import, print_function


class SquirrelError(Exception):
    pass


class NotAvailable(SquirrelError):
    pass


class ConversionError(SquirrelError):
    pass


class ToolError(Exception):
    pass


__all__ = [
    'SquirrelError',
    'ToolError',
    'NotAvailable']
