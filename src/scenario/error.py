# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------
from __future__ import absolute_import, division, print_function


class ScenarioError(Exception):
    pass


class LocationGenerationError(ScenarioError):
    pass


class CannotCreatePath(ScenarioError):
    pass


__all__ = ['ScenarioError', 'LocationGenerationError', 'CannotCreatePath']
