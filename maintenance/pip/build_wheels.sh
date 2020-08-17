#!/bin/bash

set -e

mkdir wheels_temp
for x in /opt/python/* ; do
    "$x/bin/python" -c 'import sys ; sys.exit(not (sys.version_info[:2] == (2, 7) or (3, 5, 0) <= sys.version_info <= (3, 8, 0)))' || continue
    #"$x/bin/python" -c 'import sys ; sys.exit(not (sys.version_info[:2] == (3, 7)))' || continue
    "$x/bin/pip" install --upgrade pip
    "$x/bin/pip" install --no-cache-dir -r requirements.txt
    "$x/bin/pip" wheel -v . -w wheels_temp
done

mkdir wheels
for wheel in wheels_temp/*.whl ; do
    auditwheel repair "$wheel" --plat $PLAT -w wheels
    rm "$wheel"
done
