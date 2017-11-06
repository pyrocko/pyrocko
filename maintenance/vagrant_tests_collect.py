#!/usr/bin/env python

import glob
import sys
import os
import re


def parse_result(fn, show_skips=False):
    with open(fn, 'r') as f:
        txt = f.read()

        m = re.search(r'Python Version: (Python.*)', txt)
        py_version = m.group(1)
        m = re.search(r'/test-(.*)\.py([23])\.out$', fn)
        branch = m.group(1)

        print('   python: %s' % py_version)
        print('      log: %s' % fn)
        print('      branch: %s' % branch)

        m = re.search(r'---+\nTOTAL +(.+)\n---+', txt)
        if m:
            print('      coverage: %s' % m.group(1))

        m = re.search(r'^((OK|FAILED)( +\([^\)]+\))?)', txt, re.M)
        if m:
            print('      tests: %s' % m.group(1))

        if show_skips:
            count = {}
            for x in re.findall(r'... SKIP: (.*)$', txt, re.M):
                if x not in count:
                    count[x] = 1
                else:
                    count[x] += 1

            for x in sorted(count.keys()):
                print('         skip: %s (%ix)' % (x, count[x]))

        for x in re.findall(r'^ERROR: .*$', txt, re.M):
            print('         %s' % x)

        for x in re.findall(r'^FAIL: .*$', txt, re.M):
            print('         %s' % x)


args = sys.argv[1:]

if len(args) == 0:

    boxes = os.listdir('vagrant')
    for box in boxes:
        print(box)
        results = glob.glob(os.path.join('vagrant', box, 'test-*.py[23].out'))
        if results:
            for result in results:
                parse_result(result)
        else:
            print('  ', '<no results>')
