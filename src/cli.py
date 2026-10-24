# http://pyrocko.org - GPLv3
#
# The Pyrocko Developers, 21st Century
# ---|P------/S----------~Lg----------

'''
Command line argument parsing infrastructure for Pyrocko's tools.
'''

import os
import re
import inspect
import argparse
import textwrap


def dq(x):
    return '``%s``' % x


def ldq(xs):
    return ', '.join(dq(x) for x in xs)


# Since Python 3.13, argparse may colorize its output with ANSI escape
# sequences (see ``can_colorize`` in the standard library's ``_colorize``
# module). The custom wrapping and paragraph-detection logic below predates
# this and works on the assumption that its input is plain text: it uses
# ``len()``-based wrapping (via ``textwrap``) and ``str.startswith('usage:')``
# checks to recognize the usage line, both of which are thrown off by leading
# or embedded escape sequences (e.g. the usage line then starts with
# ``\x1b[1;34m`` rather than with ``usage:``, some words get glued to escape
# sequences that count towards the wrap width although they render with zero
# width, and a wrap point can end up splitting an escape sequence or a
# colorized word in two).
#
# Colorized help is a nice feature for interactive use, so rather than
# disabling it, ``_ansi_wrap`` below reimplements the relevant bits of
# ``wrap``/``wrap_usage`` to treat ANSI SGR escape sequences as zero-width:
# it wraps a color-stripped copy of the text with the real
# ``textwrap.wrap`` (getting its exact word-splitting and whitespace
# handling for free) and then replays the result against the original,
# colorized text to reattach the escape sequences at the right positions.
#
# Colorized output is unwanted for the ``rst`` help used to build the docs
# (see ``PyrockoArgumentParser``), where stray escape sequences would end up
# in the generated ``rst`` source, so that path still asks argparse not to
# colorize in the first place (see ``_argparse_supports_color`` below).
#
# ``color`` was added as a keyword argument to ``argparse.ArgumentParser``
# and ``argparse.HelpFormatter`` in a later Python 3 version than the
# minimum supported here, hence the feature check instead of a version
# check.
_argparse_supports_color = 'color' in inspect.signature(
    argparse.ArgumentParser.__init__).parameters

_ansi_sgr_re = re.compile(r'\x1b\[[0-9;]*m')

_triple_backtick_re = re.compile(r'```(.*?)```', re.S)
_double_backtick_re = re.compile(r'``(.*?)``', re.S)

_markup_bold = '\x1b[1m'
_markup_reset = '\x1b[0m'


def _visible(s):
    return _ansi_sgr_re.sub('', s)


def _colorize_markup(s, color):
    '''
    Replace our own :literal:`\\`\\`code\\`\\`` /
    :literal:`\\`\\`\\`code\\`\\`\\`` markup (see ``PyrockoArgumentParser``)
    with its normal-``--help`` rendering: :literal:`\\`\\`code\\`\\`` becomes
    ``code``,
    :literal:`\\`\\`\\`code\\`\\`\\`` becomes ``'code'``. If ``color`` is set
    (i.e. argparse is colorizing this output, see ``format_help``), the
    replacement text is additionally set in bold, so it stands out from the
    surrounding prose the same way inline code does in the ``rst`` docs.
    '''

    if color:
        s = _triple_backtick_re.sub(
            lambda m: "%s'%s'%s" % (
                _markup_bold, m.group(1), _markup_reset), s)
        s = _double_backtick_re.sub(
            lambda m: '%s%s%s' % (
                _markup_bold, m.group(1), _markup_reset), s)
    else:
        s = _triple_backtick_re.sub(lambda m: "'%s'" % m.group(1), s)
        s = _double_backtick_re.sub(lambda m: m.group(1), s)

    return s


def _ansi_wrap(text, width, subsequent_indent=''):
    '''
    Like :py:func:`textwrap.wrap`, but treats ANSI SGR escape sequences
    embedded in ``text`` as zero-width, so that e.g. a colorized option flag
    does not count towards the wrap width and does not get split apart by a
    wrap point landing inside it.
    '''

    fragments = _ansi_sgr_re.split(text)
    codes = _ansi_sgr_re.findall(text)
    if not codes:
        return textwrap.wrap(text, width, subsequent_indent=subsequent_indent)

    plain = ''.join(fragments)

    # gaps[i]: escape sequences occurring in `text` immediately before the
    # visible character at plain[i] (gaps[len(plain)]: trailing escape
    # sequences occurring after the last visible character).
    gaps = [''] * (len(plain) + 1)
    pos = 0
    for fragment, code in zip(fragments, codes):
        pos += len(fragment)
        gaps[pos] += code

    # Locate each wrapped line's (start, end) span in `plain`.
    spans = []
    cursor = 0
    for i, line in enumerate(
            textwrap.wrap(plain, width, subsequent_indent=subsequent_indent)):

        content = line[len(subsequent_indent):] if i > 0 else line
        start = plain.index(content, cursor)
        end = start + len(content)
        spans.append((start, end))
        cursor = end

    # Build each line's colorized text. Escape sequences occurring exactly
    # at a line's end (gaps[end], e.g. a reset right after the last word of
    # that line) are attached to that line, unless the next line's span
    # starts at that very same position -- which happens when a single
    # word had to be split apart because it alone exceeded the width (no
    # whitespace was dropped in between); in that case, the escape
    # sequences are picked up as the leading gap of the next line instead,
    # so that they are emitted exactly once either way.
    lines = []
    for i, (start, end) in enumerate(spans):
        colorized = ''.join(
            gaps[j] + plain[j] for j in range(start, end))
        next_start = spans[i + 1][0] if i + 1 < len(spans) else None
        if next_start != end:
            colorized += gaps[end]
        lines.append((subsequent_indent if i > 0 else '') + colorized)

    return lines


def unwrap(s):
    if s is None:
        return None
    s = s.strip()
    parts = re.split(r'\n{2,}', s)
    lines = []
    for part in parts:
        plines = part.splitlines()
        if not any(line.startswith('  ') for line in plines):
            lines.append(' '.join(plines))
        else:
            lines.extend(plines)

        lines.append('')

    return '\n'.join(lines)


def wrap(s):
    lines = []
    parts = re.split(r'\n{2,}', s)
    for part in parts:
        plines = part.splitlines()
        if _visible(part).startswith('usage:') \
                or all(line.startswith('    ') for line in plines):
            lines.extend(plines)
        else:
            for line in plines:
                if not line:
                    lines.append(line)
                if not line.startswith(' '):
                    lines.extend(
                        _ansi_wrap(line, 79))
                else:
                    lines.extend(
                        _ansi_wrap(line, 79, subsequent_indent=' '*24))

        lines.append('')

    return '\n'.join(lines)


def wrap_usage(s):
    lines = []
    for line in s.splitlines():
        if not line.startswith('usage:'):
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, 79, subsequent_indent=' '*24))

    return '\n'.join(lines)


def formatter_with_width(n):
    class PyrockoHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def __init__(self, *args, **kwargs):
            kwargs['width'] = n
            kwargs['max_help_position'] = 24
            argparse.RawDescriptionHelpFormatter.__init__(
                self, *args, **kwargs)

            # fix alignment problems, with the post-processing wrapping
            self._action_max_length = 24

    return PyrockoHelpFormatter


class PyrockoArgumentParser(argparse.ArgumentParser):
    '''
    Tweaks and extends the standard argument parser to simplify the generation
    of the online docs.

    We want to convert the ``--help`` outputs to ``rst`` for the html docs.
    Problem is that argparse's ``HelpFormatter`` to date have no public
    interface which we could use to achieve this. The solution here is a bit
    clunky but works ok for our purposes. We allow markup like
    :literal:`\\`\\`code\\`\\`` which is kept when producing ``rst`` (by
    parsing the final ``--help`` output) but stripped out when doing normal
    ``--help``. This leads to a problem with the internal output wrapping of
    argparse which it does before the stripping. To solve, we render with
    argparse to a very wide width and do the wrapping in post-processing.
    :literal:`\\`\\`code\\`\\`` is replaced with just ``code`` in normal
    output. :literal:`\\`\\`\\`code\\`\\`\\`` is replaced with ``'code'`` in
    normal output and with :literal:`\\`\\`code\\`\\`` in ``rst`` output.
    ``rst`` output is selected with environment variable
    ``PYROCKO_RST_HELP=1``. The script ``maintenance/argparse_help_to_rst.py``
    extracts the ``rst`` help and generates the ``rst`` files for the docs.
    '''

    def __init__(
            self, prog=None, usage=None, description=None, epilog=None,
            **kwargs):

        kwargs['formatter_class'] = formatter_with_width(1000000)

        if _argparse_supports_color \
                and os.environ.get('PYROCKO_RST_HELP', '0') != '0':
            # Keep the docs free of stray escape sequences.
            kwargs.setdefault('color', False)

        description = unwrap(description)
        epilog = unwrap(epilog)

        argparse.ArgumentParser.__init__(
            self, prog=prog, usage=usage, description=description,
            epilog=epilog, **kwargs)

        if hasattr(self, '_action_groups'):
            for group in self._action_groups:
                if group.title == 'positional arguments':
                    group.title = 'Positional arguments'

                elif group.title == 'optional arguments':
                    group.title = 'Optional arguments'

                elif group.title == 'options':
                    group.title = 'Options'

        self.raw_help = False

    def format_help(self, *args, **kwargs):
        s = argparse.ArgumentParser.format_help(self, *args, **kwargs)

        # If argparse put any color into its own output (flags, metavars,
        # headers, ...), colorize our own backtick markup consistently.
        color = bool(_ansi_sgr_re.search(s))

        # replace usage with wrapped one from argparse because naive wrapping
        # does not look good.
        formatter_class = self.formatter_class
        self.formatter_class = formatter_with_width(79)
        usage = self.format_usage()
        self.formatter_class = formatter_class

        lines = []
        for line in s.splitlines():
            if _visible(line).startswith('usage:'):
                lines.append(usage)
            else:
                lines.append(line)

        s = '\n'.join(lines)

        if os.environ.get('PYROCKO_RST_HELP', '0') == '0':
            s = _colorize_markup(s, color)
            s = wrap(s)
        else:
            s = s.replace('```', '``')
            s = wrap_usage(s)

        return s


__all__ = [
    'PyrockoArgumentParser',
    'dq',
    'ldq',
]
