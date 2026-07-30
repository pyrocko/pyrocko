# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------


from pyrocko import util
from pyrocko.guts import Object, String, List

from .error import SquirrelError
from .operators.base import BaseOperator

guts_prefix = 'squirrel'


class MantraError(SquirrelError):
    pass


class Mantra(Object):
    '''
    Represents a connected graph of processing operators.
    '''

    name = String.T(default='untitled')
    operators = List.T(BaseOperator.T())

    def _raise_error(self, message):
        raise MantraError(f'{message} (Mantra {self.name})')

    def _plan_connections(self, squirrel):
        if not self.operators:
            return [], squirrel

        previous_name, previous_operator = ('squirrel', squirrel)
        by_name = {'squirrel': squirrel}
        used = set()
        connections = []
        for ioperator, operator in enumerate(self.operators):

            name = operator.name or f'operator_{ioperator}'

            if name in by_name:
                self._raise_error(f'Duplicate operator name: {name}')

            by_name[name] = operator

            if not operator.input_names:
                connections.append((operator, previous_operator))
                used.add(previous_name)

            else:
                for input_name in operator.input_names:
                    if input_name not in by_name:
                        self._raise_error(
                            f'Operator "{name}" requires input '
                            f'from yet undefined operator "{input_name}".')

                    used.add(input_name)
                    connections.append((operator, by_name[input_name]))

            previous_name, previous_operator = (name, operator)

        used.add(previous_name)

        unused = sorted(set(by_name) - used)

        if unused:
            self._raise_error(
                'Unused provider%s: %s' % (
                    util.plural_s(unused), ', '.join(unused)))

        return connections, previous_operator

    def setup(self, squirrel):
        '''
        Connect operators into a processing graph.
        '''

        connections, outlet = self._plan_connections(squirrel)

        self.outlet = outlet

        for operator in self.operators:
            operator.set_mantra(self)

        for operator, input in connections:
            operator.add_input(input)

    def describe(self):
        '''
        Get textual description of the processing graph.
        '''

        lines = []
        lines.append(f'Mantra: {self.name}\n  operators:')
        for operator in self.operators:
            lines.extend(
                f'    {line}'
                for line in operator.describe().splitlines())

        return '\n'.join(lines)


__all__ = [
    'Mantra',
    'MantraError',
]
