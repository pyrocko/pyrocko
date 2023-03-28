#!/bin/bash

set -e

rm -rf wheels_temp wheels

mkdir wheels_temp
for x in `ls /opt/python` ; do
    "/opt/python/$x/bin/python" -c 'import sys ; sys.exit(not ((3, 7, 0) <= sys.version_info) or sys.implementation.name != "cpython")' || continue
    "/opt/python/$x/bin/pip" install --upgrade pip
    "/opt/python/$x/bin/pip" wheel -v . -w wheels_temp --only-binary=:all:
done

mkdir wheels
for wheel in wheels_temp/pyrocko-*.whl ; do
    auditwheel repair "$wheel" --plat $PLAT -w dist
    rm "$wheel"
done
