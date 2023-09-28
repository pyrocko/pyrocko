#!/bin/bash

set -e

revision=$1
debianversion=$2

if [ -z "$revision" ] || [ -z "$debianversion" ]; then
    echo "usage: build_in_docker.sh <revision> <debianversion>"
    exit 1
fi

docker run -ti -v .:/src -w /src  build-deb-$debianversion  maintenance/deb/build.sh $revision
mv deb-packages deb-packages-$debianversion
