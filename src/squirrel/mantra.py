# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Processing pipeline configuration and wiring.

A :py:class:`Mantra` describes a chain (or, more generally, connected
graph) of :py:class:`~pyrocko.squirrel.operators.base.BaseOperator`
instances to be connected on top of a
:py:class:`~pyrocko.squirrel.base.Squirrel`. Mantra configurations are
Guts-serializable YAML, typically loaded via ``--mantra PATH[:PATTERN,...]``
in :app:`squirrel mantra` and other CLI tools, or constructed
programmatically and connected with :py:meth:`Mantra.setup`.
'''

from pyrocko import util
from pyrocko.guts import Object, String, List

from .error import SquirrelError
from .operators.base import BaseOperator

guts_prefix = 'squirrel'


class MantraError(SquirrelError):
    '''
    Raised on errors while planning or connecting a :py:class:`Mantra`'s
    operator graph, e.g. duplicate operator names, references to
    undefined operators, or unused operators.
    '''


class Mantra(Object):
    '''
    Represents a connected graph of processing operators.
    '''

    name = String.T(
        default='untitled',
        help='Name of this mantra, used e.g. to select it via '
             '``--mantra PATH:PATTERN`` and as a namespacing element in '
             'derived data codes.')
    operators = List.T(
        BaseOperator.T(),
        help='Operators forming the processing graph, in connection '
             'order (see :py:meth:`setup`).')

    def _raise_error(self, message):
        raise MantraError(f'{message} (Mantra {self.name})')

    def _plan_connections(self, squirrel):
        '''
        Work out which operator feeds which, without actually connecting
        them yet (see :py:meth:`setup`).

        Operators are chained implicitly, in list order: an operator with
        no ``input_names`` set takes the previous operator (or the
        squirrel, for the first one) as its sole input. An operator with
        ``input_names`` set instead takes each of the so-named,
        already-defined earlier operators as inputs (fan-in), by name.
        Every named provider (the squirrel, or an operator) must be
        consumed by something — either by a later operator, or by being
        the last operator in the list (the graph's outlet) — otherwise
        this is flagged as an error.

        :param squirrel:
            The squirrel instance to serve as the graph's root input.
        :type squirrel:
            :py:class:`~pyrocko.squirrel.base.Squirrel`

        :returns:
            ``(connections, outlet)``, where ``connections`` is a list of
            ``(operator, input)`` pairs, in an order suitable for feeding
            to successive :py:meth:`~pyrocko.squirrel.operators.base.\
BaseOperator.add_input` calls, and ``outlet`` is the final operator (or
            the squirrel itself, if there are no operators).
        :rtype:
            :py:class:`tuple`

        :raises MantraError:
            If an operator name is duplicated, an ``input_names`` entry
            refers to an undefined (or not-yet-defined) operator, or an
            operator's output is not consumed by anything.
        '''
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
        Connect operators into a processing graph on top of a squirrel.

        :param squirrel:
            The squirrel instance to serve as the graph's root input.
        :type squirrel:
            :py:class:`~pyrocko.squirrel.base.Squirrel`

        After this call, :py:attr:`outlet` holds the graph's final
        operator (or ``squirrel`` itself, if this mantra has no
        operators), ready to be queried like a squirrel (e.g.
        ``mantra.outlet.get_waveforms(...)``).
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

        :rtype:
            str

        Requires :py:meth:`setup` to have been called first.
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
