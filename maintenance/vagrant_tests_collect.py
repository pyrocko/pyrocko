#!/usr/bin/env python3

import glob
import os
import re

from pyrocko.guts import Object, String, Dict, List


class TestResult(Object):
    package = String.T()
    branch = String.T(optional=True)
    box = String.T()
    py_version = String.T(optional=True)
    prerequisite_versions = Dict.T(
        String.T(), String.T(), optional=True, default={})
    log = String.T(optional=True, yamlstyle='|')
    result = String.T(optional=True)
    errors = List.T(String.T(), optional=True, default=[], yamlstyle='block')
    fails = List.T(String.T(), optional=True, default=[], yamlstyle='block')
    skips = List.T(String.T(), optional=True, default=[], yamlstyle='block')


def parse_result(res, package, box, fn):

    with open(fn, 'r') as f:
        txt = f.read()

        lines = txt.splitlines()
        for line in lines[:7]:
            pack, vers = line.split(': ')
            res.prerequisite_versions[pack] = vers

        txt = '\n'.join(
            line for line in lines if not re.match(r'^Q\w+::', line))
        txt = re.sub(r' +\n', '\n', txt)

        res.log = txt.strip()

        m = re.search(r'---+\nTOTAL +(.+)\n---+', txt)
        if m:
            res.coverage = m.group(1)

        m = re.search(r'^((OK|FAILED)( +\([^\)]+\))?)', txt, re.M)
        if m:
            res.result = m.group(1)

        count = {}
        for x in re.findall(r'... SKIP: (.*)$', txt, re.M):
            if x not in count:
                count[x] = 1
            else:
                count[x] += 1

        for x in sorted(count.keys()):
            res.skips.append('%s (%ix)' % (x, count[x]))

        for x in re.findall(r'^ERROR: .*$', txt, re.M):
            res.errors.append(x)

        for x in re.findall(r'^FAIL: .*$', txt, re.M):
            res.fails.append(x)


def iter_results():
    package = 'pyrocko'
    if os.path.exists('vagrant'):
        boxes = os.listdir('vagrant')

    else:
        boxes = [os.path.basename(os.path.abspath('.'))]
        os.chdir('../..')

    for box in sorted(boxes):

        fns = glob.glob(os.path.join('vagrant', box, 'test-*.py[23].out'))
        if fns:
            for fn in fns:
                m = re.search(r'/test-(.*)\.py([23])\.out$', fn)
                res = TestResult(package=package, branch=m.group(1), box=box)
                res.py_version = m.group(2)
                parse_result(res, package, box, fn)
                yield res

        else:
            res = TestResult(
                package=package, box=box,
                result='ERROR (running the tests failed)')

            yield res


if __name__ == '__main__':
    for r in iter_results():
        print(r)
