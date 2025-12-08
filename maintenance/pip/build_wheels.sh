#!/bin/bash

set -e

rm -rf wheels_temp wheels

mkdir wheels_temp

pvs="
cp310-cp310
cp311-cp311
cp312-cp312
cp313-cp313
cp314-cp314
"

for pv in $pvs ; do
    echo "===================================================================="
    echo "Building for ${pv}"
    echo "===================================================================="
    "/opt/python/$pv/bin/pip" install --upgrade pip
    "/opt/python/$pv/bin/pip" wheel -v . -w wheels_temp --only-binary=:all:
done

mkdir wheels
for wheel in wheels_temp/pyrocko-*.whl ; do
    auditwheel repair "$wheel" --plat $PLAT -w dist
    rm "$wheel"
done

echo "===================================================================="
echo "Wheels in ./dist:"
ls dist
echo "===================================================================="
