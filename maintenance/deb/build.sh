#!/bin/bash

set -e

revision=$1
if [ -z "$revision" ]; then
    echo "usage: build.sh <revision>"
    exit 1
fi

rm -rf  dist
python3 setup.py sdist
archive=`find dist -name 'pyrocko-*.tar.gz'`
version=${archive/dist\/pyrocko-/}
version=${version/.tar.gz/}
echo "Archive: $archive"
echo "Version: $version"

debmake -r $revision -a $archive -b":python3" -i debuild
mkdir -p deb-packages
mv pyrocko_$version* deb-packages
mv pyrocko-dbgsym_$version* deb-packages
mv pyrocko-$version.tar.gz deb-packages
