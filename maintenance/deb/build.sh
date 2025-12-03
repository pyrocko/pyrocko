#!/bin/bash

set -e

revision=$1
debianversion=$2

if [ -z "$revision" ] || [ -z "$debianversion" ]; then
    echo "usage: build.sh <revision> <debianversion>"
    exit 1
fi

rm -rf  dist
python3 setup.py sdist
archive=`find dist -name 'pyrocko-*.tar.gz'`
version=${archive/dist\/pyrocko-/}
version=${version/.tar.gz/}
echo "Archive: $archive"
echo "Version: $version"

targetdir=deb-packages-$debianversion
rm -rf $targetdir

rm -rf build-deb-${debianversion}
mkdir build-deb-${debianversion}
cd build-deb-${debianversion}

cp -r ../dist .

debmake -r $revision -z tar.gz -a $archive -b":python3" -i debuild
mkdir -p $targetdir
mv pyrocko_$version* $targetdir
mv pyrocko-dbgsym_$version* $targetdir
mv pyrocko-$version.tar.gz $targetdir
mv $targetdir ..
