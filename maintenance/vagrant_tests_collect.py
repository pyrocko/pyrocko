#!/usr/bin/env python

from __future__ import print_function

import glob
import os
import re


def parse_result(fn, show_skips=False):
    lines = []
    with open(fn, 'r') as f:
        txt = f.read()

        versions = txt.splitlines()[:7]

        m = re.search(r'/test-(.*)\.py([23])\.out$', fn)
        branch = m.group(1)
        py_version = m.group(2)
        lines.append('  running under Py %s' % py_version)

        lines.append('    versions:')
        for version in versions:
            lines.append('       %s' % version)

        lines.append('    log: %s' % fn)
        lines.append('    branch: %s' % branch)

        m = re.search(r'---+\nTOTAL +(.+)\n---+', txt)
        if m:
            lines.append('    coverage: %s' % m.group(1))

        m = re.search(r'^((OK|FAILED)( +\([^\)]+\))?)', txt, re.M)
        if m:
            lines.append('    tests: %s' % m.group(1))

        if show_skips:
            count = {}
            for x in re.findall(r'... SKIP: (.*)$', txt, re.M):
                if x not in count:
                    count[x] = 1
                else:
                    count[x] += 1

            for x in sorted(count.keys()):
                lines.append('         skip: %s (%ix)' % (x, count[x]))

        for x in re.findall(r'^ERROR: .*$', txt, re.M):
            lines.append('         %s' % x)

        for x in re.findall(r'^FAIL: .*$', txt, re.M):
            lines.append('         %s' % x)

    return lines


def iter_results():
    if os.path.exists('vagrant'):
        boxes = os.listdir('vagrant')

    else:
        boxes = [os.path.basename(os.path.abspath('.'))]
        os.chdir('../..')

    for box in boxes:
        lines = []
        lines.append(box)
        results = glob.glob(os.path.join('vagrant', box, 'test-*.py[23].out'))
        if results:
            for result in results:
                lines.extend(parse_result(result))
        else:
            lines.append('  ', '<no results>')

        lines.append('')

        yield '\n'.join(lines)


if __name__ == '__main__':
    for r in iter_results():
        print(r)
