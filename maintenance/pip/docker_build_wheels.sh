#!/bin/bash

set -e

if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "repos not clean"
    exit 1
fi

wd=`pwd`

plat=manylinux2010_x86_64 

srcdir="$wd/pyrocko"
outdir="$wd/wheels"
branch=`git rev-parse --abbrev-ref HEAD`

rm -rf $outdir
mkdir $outdir

rm -rf $srcdir
git clone -b $branch ../.. pyrocko
echo "creating pip wheels for branch $branch"

sudo docker run  \
    --mount type=bind,source="$srcdir",destination=/src \
    --mount type=bind,source="$outdir",destination=/wheels \
    --env PLAT=$plat \
    quay.io/pypa/$plat \
    /src/maintenance/pip/build_wheels.sh

sudo chown -R `id -u` wheels
sudo chgrp -R `id -g` wheels
