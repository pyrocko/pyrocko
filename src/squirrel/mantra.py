# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------


from pyrocko.guts import Object, String, List

from .operators.base import BaseOperator

guts_prefix = 'squirrel'


class Mantra(Object):
    name = String.T(default='untitled')
    operators = List.T(BaseOperator.T())

    def setup(self, sq, arrays=None):
        if not self.operators:
            self.outlet = sq
        else:
            operators = list(self.operators)
            previous = sq
            while operators:
                operator = operators.pop(0)
                operator.set_input(previous)
                if arrays is not None:
                    if hasattr(operator, 'set_arrays'):
                        operator.set_arrays(arrays)

                previous = operator

            self.outlet = operator

    def print_operator_mappings(self):
        print()
        print('Mantra: %s' % self.name)
        for operator in self.operators:
            print(operator.describe())

        print()


__all__ = [
    'Mantra',
]
