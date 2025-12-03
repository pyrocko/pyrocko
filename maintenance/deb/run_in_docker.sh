#!/bin/bash

set -e

revision=$1
debianversion=$2
debianversion_all="debian-12,debian-13,ubuntu-22.04,ubuntu-24.04"

if [ -z "$revision" ] || [ -z "$debianversion" ]; then
    echo "usage: run_in_docker.sh <revision> <debianversion>"
    echo "    <debianversion>: $debianversion_all" 
    echo "                     or 'all'"
    echo "    Built deb packages and docker images must be available."
    exit 1
fi

if [[ $debianversion == 'all' ]] ; then
    debianversion=$debianversion_all
fi

IFS=',' read -ra debianversions <<<"$debianversion"

result=""

for dv in "${debianversions[@]}"; do
    image="pyrocko-fat-nest-$dv"
    echo -e "\u250f\u2501\u2501 BEGIN running in $image \n\u2503"
    if [ -z "$DISPLAY" ] ; then
        x_stuff=""
    else
        x_stuff="-v $XAUTHORITY:/tmp/.XAuthority -e XAUTHORITY=/tmp/.XAuthority -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
    fi
    docker run \
        --rm \
        -ti \
        -v .:/src \
        -w /src \
        $x_stuff \
        $image maintenance/deb/run.sh $revision $dv ${@:3} \
        && { result="$result\n$dv: ok" ; echo -e "\u2503\n\u2517\u2501\u2501 END running in $image \n" ; } \
        || { result="$result\n$dv: failed" ; echo -e "\u2503\n\u2517\u2501\u2501 FAILURE running in $image \n" ; }

        # may help if problems occur:
        # --net host \
        #-e QT_X11_NO_MITSHM=1 \
        #-e LIBGL_ALWAYS_SOFTWARE=1 \
        #-e LIBGL_ALWAYS_INDIRECT=1 \
        #-e __GLX_VENDOR_LIBRARY_NAME=mesa \

done

echo -e "$result"
