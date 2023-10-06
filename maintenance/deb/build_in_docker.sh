#!/bin/bash

set -e

revision=$1
debianversion=$2
debianversion_all="debian-10,debian-11,debian-12,ubuntu-20.04,ubuntu-22.04"

if [ -z "$revision" ] || [ -z "$debianversion" ]; then
    echo "usage: build_in_docker.sh <revision> <debianversion>"
    echo "    <debianversion>: $debianversion_all"
    echo "                     or 'all'"
    echo "    Respective docker images must be available, see"
    echo "    docker/build_docker_images.sh"
    exit 1
fi

if [[ $debianversion == 'all' ]] ; then
    debianversion=$debianversion_all
fi

IFS=',' read -ra debianversions <<<"$debianversion"

for dv in "${debianversions[@]}"; do
    docker run --rm -v .:/src -w /src  pyrocko-build-deb-$dv  maintenance/deb/build.sh $revision $dv
done
