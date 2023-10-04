#!/bin/bash

# run 'docker system prune -a' for a clean start

set -e

rm -rf pyrocko-test-data
wget -r http://data.pyrocko.org/testing/pyrocko/ -nv --no-parent -nH --cut-dirs=2 -P pyrocko-test-data

docker build nest -t pyrocko-nest

rm -rf fat-nest/pyrocko-test-data
cp -r pyrocko-test-data fat-nest
docker build fat-nest -t pyrocko-fat-nest

docker build docs -t pyrocko-docs
docker build util -t pyrocko-util

rm -rf pyrocko/pyrocko.git
git clone --bare .. pyrocko/pyrocko.git
docker build pyrocko -t pyrocko

for x in build-deb-* ; do
    docker build $x -t $x
done

for x in test-deb-* ; do
    docker build $x -t $x
done
