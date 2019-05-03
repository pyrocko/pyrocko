#!/bin/bash

set -e

cd anaconda-build-boxes

boxes="$1"
if [ -z "$boxes" ]; then
    boxes=`ls`
fi

for box in $boxes; do
    cd $box
    echo "Building Anaconda packages on $box"
    ./outside.sh
    cd ..
done
