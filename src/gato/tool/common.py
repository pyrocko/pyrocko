# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Gato command line tool utilities.
'''

from pyrocko.util import glob_filter, GlobFilterNoMatch

from pyrocko.gato.array import \
    get_named_arrays, SensorArray
from pyrocko.gato.error import GatoToolError
from pyrocko.gato.io import load_all


def add_sensor_array_arguments(parser):

    parser.add_argument(
        '--arrays',
        dest='arrays',
        metavar='PATH[:PATTERN,...]|:NAME', nargs='+',
        default=[],
        help='Select sensor arrays to be used. Definitions are read from YAML '
             'configuration file ``PATH`` and selected if their name matches '
             'any of the given ``PATTERN``. If no patterns are given, all '
             'arrays defined in the file are used. Builtin arrays can be '
             'selected by ``:NAME``, e.g. ``:norsar``. Multiple definitions '
             'may be specified. Example: '
             '``--arrays :grf :norsar custom.array.yaml:test1`` would select '
             'built-in arrays ``:grf`` and ``:norsar``, and array ``test1`` '
             'from file ``custom.array.yaml``.')


def _load_sensor_arrays(path, patterns):
    try:
        arrays = list(load_all(path, want=SensorArray))
    except Exception as e:
        raise GatoToolError(
            'Error while loading SensorArray definitions from file "%s": %s'
            % (path, str(e)))

    for array in arrays:
        array.set_defined_in(path)

    names = []
    for array in arrays:
        if array.name.startswith(':'):
            raise GatoToolError(
                'Invalid array name: "%s". Names starting with ":" '
                'are reserved for built-in arrays.' % array.name)

        if array.name in names:
            raise GatoToolError(
                'Duplicate array name in file "%s": %s' % (
                    path, array.name))

        names.append(array.name)

    if not patterns:
        return arrays

    try:
        matching_names = set(glob_filter(
            patterns, names, raise_if_nomatch=True))

    except GlobFilterNoMatch as e:
        raise GatoToolError(str(e)) from None

    return [array for array in arrays if array.name in matching_names]


def _sensor_arrays_from_spec(spec):
    spec = spec.strip()
    if spec == ':':
        return get_named_arrays()

    if spec.startswith(':'):
        # make sure all names start with ':' so that ':grf,:norsar' and
        # ':grf,norsar' are equivalent
        names = [':' + name.strip().lstrip(':') for name in spec.split(',')]
        return get_named_arrays(names)

    if ':' in spec:
        path, spec_patterns = spec.split(':', maxsplit=1)
        patterns = [
            pattern.strip()
            for pattern in spec_patterns.split(',')
            if pattern.strip()]
    else:
        path, patterns = spec, []

    path = path.strip()

    return _load_sensor_arrays(path, patterns)


def sensor_arrays_from_arguments(args, check_have_arrays=True):
    arrays = {}
    for spec in args.arrays:
        for array in _sensor_arrays_from_spec(spec):
            if array.name in arrays:
                raise GatoToolError(
                    'Duplicate array name: %s\n'
                    '  defined in: %s\n'
                    '  also defined in: %s' % (
                        array.name,
                        arrays[array.name].get_defined_in(),
                        array.get_defined_in()))

            arrays[array.name] = array

    if not arrays and check_have_arrays:
        raise GatoToolError(
            'No sensor arrays added or no matching arrays found. Use '
            '--arrays.\nExamples:\n'
            '  --arrays :grf :norsar           Use built-in arrays `:grf` '
            'and `:norsar`.\n'
            '  --arrays :                      Select all built-ins.\n'
            '  --arrays my.array.yaml          Select all arrays defined in '
            '`my.array.yaml`.\n'
            '  --arrays my.array.yaml:a2,a3    Select arrays `a2` and `a3` '
            'from `my.array.yaml`.')

    return list(arrays.values())


__all__ = [
    'add_sensor_array_arguments',
    'sensor_arrays_from_arguments',
]
