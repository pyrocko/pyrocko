#!/bin/bash

set -e

if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "repos not clean"
    exit 1
fi

wd=`pwd`

srcdir="$wd/pyrocko_src_data"
wheeldir="$wd/wheels"
branch=`git rev-parse --abbrev-ref HEAD`

rm -rf "$srcdir"
git clone -b $branch ../.. $srcdir
cp -r ../../test/data $srcdir/test/
cp -r ../../test/example_run_dir $srcdir/test/


for image in python:3.7-buster ; do
    volume="pyrocko_src_data_${image/:/_}"

    sudo docker volume create $volume
    sudo docker run --rm -v "$srcdir":/src -v $volume:/dst busybox cp -r /src /dst

    sudo docker run \
        --mount source=$volume,destination=/src \
        --mount type=bind,source="$wheeldir",destination=/wheels \
        $image \
        /src/maintenance/pip/test_wheels.sh

    sudo docker volume rm $volume
done

rm $srcdir
