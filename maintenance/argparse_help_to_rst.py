
import os
import io
import re
import sys
import importlib
import contextlib


def parse_subcommands(part):
    out = ['Subcommands:']
    subcommand = None
    description = None
    for line in part.splitlines()[2:]:
        if line.startswith('    ') and line[4] != ' ':
            if subcommand:
                out.append((subcommand, description))

            words = line.split(None, 1)
            subcommand = words.pop(0)
            description = words[0] if words else ''
        else:
            if subcommand is not None:
                description += ' ' + line.strip()

    out.append((subcommand, description))
    return out


def parse_usage(subpath, s):
    assert s.lower().startswith('usage: ')
    words = s[len('usage: '):].split(None, 1 + len(subpath))
    assert words[1:-1] == subpath
    return ' '.join(words[:-1]), s


def parse_group(part):
    lines = part.splitlines()
    out = [lines[0]]
    arg = None
    description = None
    for line in lines[1:]:
        if line.startswith('  ') and line[2] != ' ':
            if arg:
                out.append((arg, description))

            words = re.split(r' {2,}', line.strip(), 1)
            arg = words.pop(0)
            description = words[0] if words else ''

        else:
            if arg is not None:
                description += ' ' + line.strip()

    out.append((arg, description))
    return out


def parse_help(subpath, s):
    d = {}
    parts = re.split(r'\n{2,}', s)
    program, usage = parse_usage(subpath, parts[0])
    d['program'] = program
    d['usage'] = usage
    d['headline'] = \
        parts[1] \
        if parts[1].find('\n') == -1 and parts[1].endswith('.') \
        else ''
    d['subcommands'] = []
    d['parts'] = []

    for part in parts[2 if d['headline'] else 1:]:
        lines = part.splitlines()
        if lines[0].lower() == 'subcommands:':
            d['subcommands'] = parse_subcommands(part)
            d['parts'].append(('subcommands', d['subcommands']))
        elif lines[0].endswith(':') \
                and all(line.startswith('  ') for line in lines[1:]):
            d['parts'].append(('arg_group', parse_group(part)))
        else:
            d['parts'].append(('text', part))

    return d


class Exit(Exception):
    pass


def myexit(state):
    raise Exit


def capture(func, *args):
    realexit = sys.exit
    sys.exit = myexit
    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            func(*args)

    except Exit:
        pass

    finally:
        sys.exit = realexit

    return f.getvalue()


def format_title(ul, s):
    return [s, ul * len(s), '']


def indent(n, lines):
    return [' ' * n + line if line else '' for line in lines]


def format_usage(program, s):
    return ['.. application:: %s' % program, ''] \
        + indent(0, ['.. code-block:: none', ''] + indent(4, s.splitlines()))


def format_text(program, s):
    lines = s.splitlines()
    if all(line.startswith('    ') for line in lines):
        return ['.. code-block:: none', ''] + lines + ['']
    else:
        return [' '.join(lines), '']


def format_arg_group(program, part):
    part = list(part)
    lines = ['**%s**' % part.pop(0), '']
    for arg, description in part:
        lines.extend([
            '.. describe:: %s' % arg, '', '    %s' % description, ''])

    return lines


def format_subcommand(program, part):
    part = list(part)
    lines = ['**%s**' % part.pop(0), '']
    lines.extend(['.. toctree::', '    :maxdepth: 1', ''])
    for arg, description in part:
        lines.append('    %s %s - %s<%s>' % (
            program, arg,
            description,
            program.replace(' ', '_') + '_' + arg))
    return lines


def format_part(kind, program, part):
    return {
        'text': format_text,
        'arg_group': format_arg_group,
        'subcommands': format_subcommand}[kind](program, part)


def format_rst(dhelp):
    lines = []
    a = lines.append
    e = lines.extend

    a('Command reference')
    a('')
    e(format_title('-', '``%s``' % dhelp['program']))
    a('')
    a('.. program:: %s' % dhelp['program'])
    a('')
    a(dhelp['headline'])
    a('')
    e(format_usage(dhelp['program'], dhelp['usage']))

    a('')

    for kind, part in dhelp['parts']:
        e(format_part(kind, dhelp['program'], part))

    return '\n'.join(lines)


def make_rst(path, main, subpath=[]):
    shelp = capture(main, subpath + ['--help'])
    dhelp = parse_help(subpath, shelp)

    fn = os.path.join(path, dhelp['program'].replace(' ', '_') + '.rst')
    with open(fn, 'w') as f:
        f.write(format_rst(dhelp))

    for subcommand, _ in dhelp['subcommands'][1:]:
        make_rst(path, main, subpath + [subcommand])


os.environ['PYROCKO_RST_HELP'] = '1'
py_path, out_path = sys.argv[1:3]

mod_name, func_name = py_path.rsplit('.', 1)

main = getattr(importlib.import_module(mod_name), func_name)
if not os.path.exists(out_path):
    os.mkdir(out_path)

make_rst(out_path, main)
