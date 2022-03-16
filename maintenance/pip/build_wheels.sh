#!/bin/bash

set -e

rm -rf wheels_temp wheels

mkdir wheels_temp
mv pyproject.toml pyproject.toml.orig
for x in `ls /opt/python` ; do
    "/opt/python/$x/bin/python" -c 'import sys ; sys.exit(not ((3, 6, 0) <= sys.version_info < (3, 11, 0)))' || continue
    [ -f "maintenance/pip/pyproject-build-pip-$x.toml" ] || continue
    cp "maintenance/pip/pyproject-build-pip-$x.toml" pyproject.toml
    "/opt/python/$x/bin/pip" install --upgrade pip
    "/opt/python/$x/bin/pip" install --only-binary=:all: --no-cache-dir -r "maintenance/pip/requirements-build-pip-$x.txt"
    "/opt/python/$x/bin/pip" wheel -v . -w wheels_temp --only-binary=:all:
done

mkdir wheels
for wheel in wheels_temp/pyrocko-*.whl ; do
    auditwheel repair "$wheel" --plat $PLAT -w wheels
    rm "$wheel"
done

mv pyproject.toml.orig pyproject.toml
