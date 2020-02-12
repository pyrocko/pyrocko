#!/bin/bash

set -e


cd /src

mkdir wheels

ls /opt/python

for x in /opt/python/* ; do
    "$x/bin/python" -c 'import sys ; sys.exit(not (sys.version_info[:2] == (2, 7) or sys.version_info >= (3, 5, 0)))' || continue
    "$x/bin/pip" install --no-cache-dir -r requirements.txt
    "$x/bin/pip" wheel -v . -w wheels
done

for wheel in wheels/*.whl ; do 
    auditwheel repair "$wheel" --plat $PLAT -w /wheels
    rm "$wheel"
done
