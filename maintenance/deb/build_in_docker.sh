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

result=""

for dv in "${debianversions[@]}"; do
    image=pyrocko-build-deb-$dv
    echo -e "\u250f\u2501\u2501 BEGIN building in $image \n\u2503"
    docker run --rm -v .:/src -w /src  $image  maintenance/deb/build.sh $revision $dv \
        && { result="$result\n$dv: ok" ; echo -e "\u2503\n\u2517\u2501\u2501 END building in $image \n" ; } \
        || { result="$result\n$dv: failed" ; echo -e "\u2503\n\u2517\u2501\u2501 FAILURE building in $image \n" ; }
done

echo -e "$result"
