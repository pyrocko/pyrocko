
import sys
import os
import re
import platform
import argparse
import shlex
import subprocess
import textwrap
import sysconfig


def is_virtual_environment():
    return sys.base_prefix != sys.prefix or hasattr(sys, "real_prefix")


def externally_managed_path():
    # https://peps.python.org/pep-0668/

    try:
        scheme = sysconfig.get_default_scheme()
    except AttributeError:
        scheme = sysconfig._get_default_scheme()

    return os.path.join(
        sysconfig.get_path('stdlib', scheme),
        'EXTERNALLY-MANAGED')


def is_externally_managed():
    try:
        return not is_virtual_environment() \
            and os.path.exists(externally_managed_path())
    except Exception:
        return False


def wrap(s):
    lines = []
    parts = re.split(r'\n{2,}', s)
    for part in parts:
        if part.startswith('usage:'):
            lines.extend(part.splitlines())
        else:
            for line in part.splitlines():
                if not line:
                    lines.append(line)
                if not line.startswith(' '):
                    lines.extend(
                        textwrap.wrap(line, 79,))
                else:
                    lines.extend(
                        textwrap.wrap(line, 79, subsequent_indent=' '*24))

        lines.append('')

    return '\n'.join(lines)


def yes(parser):
    parser.add_argument(
        '-y', '--yes',
        help='do not ask any questions (batch mode)',
        action='store_true',
        default=False)


def quiet(parser):
    parser.add_argument(
        '-q', '--quiet',
        help='do not print executed commands',
        action='store_true',
        default=False)


commands = {
    'description': wrap(
        'This is Pyrocko\'s "from source" installation helper.\n\nIt provides '
        'shortcut commands to circumvent pip\'s default automatic dependency '
        'resolution which may be problematic, e.g. when prerequisites should '
        'be supplied either by the system\'s native package manager, or by '
        'other non-pip package managers, like conda. '
        '\n\n'
        'Examples:\n\n'
        'For a system-wide installation of Pyrocko "from source", run:'
        '\n\n'
        '    /usr/bin/python install.py deps system\n'
        '    /usr/bin/python install.py system\n'
        '\n'
        'For installation "from source" into the currently activated conda '
        'environment:\n\n'
        '    python install.py deps conda\n'
        '    python install.py user\n'
        '\n'
        'For installation "from source" into a fresh venv with prerequisites '
        'supplied by the systems native package manager:\n\n'
        '    /usr/bin/python -m venv --system-site-packages myenv\n'
        '    source myenv/bin/activate\n'
        '    python install.py deps system\n'
        '    python install.py user\n'
        '\n'
        'For installation "from source" into a fresh venv with prerequisites '
        'supplied by pip (result is in this case similar to a standard '
        '"pip install .").:\n\n'
        '    /usr/bin/python -m venv myenv\n'
        '    source myenv/bin/activate\n'
        '    python install.py deps pip\n'
        '    python install.py user\n'
        '\n'
        'For batch installations with no questions add --yes --quiet to the '
        'selected subcommands.'
        ),
    'subcommands': {
        'deps': {
            'help': 'install prerequisites',
            'description': wrap(
                'Install Pyrocko\'s prerequisites for a subsequent build and '
                'install "from source", using the selected installation type. '
                'Please consult the --help message of the available '
                'subcommands for further information.'),
            'subcommands': {
                'pip': {
                    'help': 'install prerequisites using pip',
                    'description': wrap(
                        'Install Pyrocko\'s prerequisites using pip into user '
                        'environment. This command invokes `pip install` with '
                        'the appropriate list of prerequisites to prepare the '
                        'user\'s environment for a subsequent build and '
                        'install of Pyrocko "from source".'),
                    'arguments': [yes, quiet],
                },
                'conda': {
                    'help': 'install prerequisites using conda',
                    'description': wrap(
                        'Install Pyrocko\'s prerequisites using conda into '
                        'into the user\'s environment. This command invokes '
                        '`conda install` with the appropriate list of '
                        'prerequisites to prepare the currently selected '
                        'conda environment for a subsequent build and install '
                        'of Pyrocko "from source".'),
                    'arguments': [yes, quiet],
                },
                'system': {
                    'help': 'install prerequisites using the system\'s '
                            'package manager',
                    'description': wrap(
                        'Install prerequisites using the system\'s '
                        'package manager. On supported platforms, this '
                        'command invokes the system\'s package manager '
                        'with the appropriate list of system packages to '
                        'prepare for a subsequent build and install of '
                        'Pyrocko "from source".'),
                    'arguments': [yes, quiet],
                },
            },
        },
        'user': {
            'help': 'install into user or conda environment',
            'description': wrap(
                'Build Pyrocko "from source" and install it into user or '
                'conda environment. Use this installation method if you do '
                'not have "sudo" access on your system, or if you want to '
                'install into a virtual, or a conda environment. The selected '
                'options will prevent pip from automatically installing '
                'dependencies. Use one of the `deps` subcommands to satisfy '
                'Pyrocko\'s requirements before running this installer.'),
            'arguments': [yes, quiet],
        },
        'system': {
            'help': 'install system-wide',
            'description': wrap(
                'Build Pyrocko "from source" and install it system-wide for '
                'all users. Requires "sudo" access. The selected options will '
                'prevent pip from automatically installing dependencies. Use '
                'the "deps system" subcommand to satisfy ' 'Pyrocko\'s '
                'requirements with system packages before running this '
                'installer.'),
            'arguments': [yes, quiet],
        },
    }
}


def die(message):
    sys.exit('Error: %s' % message)


def confirm(s, force, quiet):
    if not force:
        try:
            return input(
                'Execute:\n\n%s\n\nProceed? [y/n] ' % s).lower() == 'y'
        except KeyboardInterrupt:
            print()
            return False

    elif not quiet:
        print('Running:\n\n%s\n\n' % s)
        return True
    else:
        return True


def do_command(cmd, force=False, quiet=False):
    qcmd = indent(' '.join(shlex.quote(s) for s in cmd))

    if confirm(qcmd, force, quiet):
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            die('Error: Called process exited with error.')

        except OSError as e:
            die('Could not run the requested command: %s' % e)
    else:
        sys.exit('Aborted.')


def indent(s):
    return '\n'.join('  ' + line for line in s.splitlines())


def do_shell_script(fn, force=False, quiet=False):
    qscript = indent(open(fn, 'r').read())

    if confirm(qscript, force, quiet):
        os.execl('/bin/sh', 'sh', fn)
    else:
        sys.exit('Aborted.')


def cmd_deps_conda(parser, args):

    if platform.system().lower() == 'windows':
        requirements = 'requirements-conda-windows.txt'
    else:
        requirements = 'requirements-conda.txt'

    do_command(
        ['conda', 'install', '--file', requirements],
        force=args.yes, quiet=args.quiet)


def cmd_deps_pip(parser, args):
    python = sys.executable or 'python3'
    do_command(
        [python, '-m', 'pip', 'install', '-r', 'requirements-all.txt'],
        force=args.yes, quiet=args.quiet)


def cmd_deps_system(parser, args):
    distribution = ''
    try:
        distribution = platform.linux_distribution()[0].lower().rstrip()
    except Exception:
        pass

    if not distribution:
        try:
            uname = platform.uname()
            if uname[2].find('arch') != -1:
                distribution = 'arch'
            elif uname[3].lower().find('ubuntu') != -1:
                distribution = 'ubuntu'
            elif uname[3].lower().find('debian') != -1:
                distribution = 'debian'
        except Exception:
            pass

    if not distribution:
        try:
            with open('/etc/redhat-release', 'r') as f:
                x = f.read()
                if x:
                    distribution = 'rpm'

        except Exception:
            pass

    if not distribution:
        sys.exit(
            'Cannot determine platform for automatic prerequisite '
            'installation.')

    if distribution == 'ubuntu':
        distribution = 'debian'

    if distribution.startswith('centos'):
        distribution = 'centos'

    fn = 'prerequisites/prerequisites_%s_python%i.sh' % (
            distribution, sys.version_info.major)

    do_shell_script(fn, force=args.yes, quiet=args.quiet)


def cmd_user(parser, args):
    python = sys.executable or 'python3'
    do_command([
        python, '-m', 'pip', 'install',
        '--no-deps',
        '--no-build-isolation',
        '--force-reinstall',
        '--upgrade',
        '.'],
        force=args.yes, quiet=args.quiet)


def cmd_system(parser, args):
    python = sys.executable or 'python3'
    pip_cmd = [
        'sudo', python, '-m', 'pip', 'install',
        '--no-deps',
        '--no-build-isolation',
        '--force-reinstall',
        '--upgrade']

    if is_externally_managed():
        pip_cmd.append('--break-system-packages')

    pip_cmd.append('.')

    do_command(pip_cmd, force=args.yes, quiet=args.quiet)


def print_help(parser, args):
    parser.print_help()


def kwargs(d, keys):
    return dict((k, d[k]) for k in keys if k in d)


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs):
        kwargs['width'] = 79
        argparse.RawDescriptionHelpFormatter.__init__(
            self, *args, **kwargs)


def make_parser(d, name=None, parent=None, path=()):

    if parent is None:
        parser = argparse.ArgumentParser(
            formatter_class=HelpFormatter,
            **kwargs(d, ['description']))
        parser.set_defaults(func=print_help, parser=parser)

    else:
        parser = parent.add_parser(
            name,
            formatter_class=HelpFormatter,
            **kwargs(d, ['help', 'description']))

    if 'arguments' in d:
        for func in d['arguments']:
            func(parser)

    if 'subcommands' in d:
        subparsers = parser.add_subparsers(title='subcommands')
        for name, d in d['subcommands'].items():
            subparser = make_parser(d, name, subparsers, path + (name,))
            subparser.set_defaults(
                func=globals().get(
                    'cmd_%s' % '_'.join(path + (name,)),
                    print_help),
                parser=subparser)

    return parser


parser = make_parser(commands)
args = parser.parse_args()
args.func(args.parser, args)
