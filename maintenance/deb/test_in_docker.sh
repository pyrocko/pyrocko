#!/bin/bash

set -e

revision=$1
debianversion=$2
debianversion_all="debian-10,debian-11,debian-12,ubuntu-20.04,ubuntu-22.04"

if [ -z "$revision" ] || [ -z "$debianversion" ]; then
    echo "usage: test_in_docker.sh <revision> <debianversion>"
    echo "    <debianversion>: $debianversion_all"
    echo "                     or 'all'"
    echo "    Built deb packages and docker images must be available."
    exit 1
fi

if [[ $debianversion == 'all' ]] ; then
    debianversion=$debianversion_all
fi

IFS=',' read -ra debianversions <<<"$debianversion"

for dv in "${debianversions[@]}"; do
    image="pyrocko-fat-nest-$dv"
    echo -e "\n\u2699\u2699\u2699 Running in $image \u2699\u2699\u2699\n"
    docker run \
        --rm \
        -ti \
        -v .:/src \
        -w /src \
        $image maintenance/deb/test.sh $revision $dv ${@:3}
        # may help if problems occur:
        #-e QT_X11_NO_MITSHM=1 \
        #-e LIBGL_ALWAYS_SOFTWARE=1 \
        #-e LIBGL_ALWAYS_INDIRECT=1 \
        #-e __GLX_VENDOR_LIBRARY_NAME=mesa \
done
