#!/bin/bash

# run 'docker system prune -a' for a clean start

set -e

rm -rf pyrocko-test-data
wget -r http://data.pyrocko.org/testing/pyrocko/ -nv --no-parent -nH --cut-dirs=2 -P pyrocko-test-data

docker build nest -t pyrocko-nest --build-arg base_image=debian:stable

rm -rf fat-nest/pyrocko-test-data
cp -r pyrocko-test-data fat-nest
docker build fat-nest -t pyrocko-fat-nest --build-arg base_image=pyrocko-nest

docker build docs -t pyrocko-docs
docker build util -t pyrocko-util

rm -rf pyrocko/pyrocko.git
git clone --bare .. pyrocko/pyrocko.git
docker build pyrocko -t pyrocko

for version in 10 11 12 ; do
    docker build nest -t pyrocko-nest-debian-$version --build-arg base_image=debian:$version
    docker build fat-nest -t pyrocko-fat-nest-debian-$version --build-arg base_image=pyrocko-nest-debian-$version
    docker build build-deb -t pyrocko-build-deb-debian-$version --build-arg base_image=debian:$version
    docker build test-deb -t pyrocko-test-deb-debian-$version --build-arg base_image=debian:$version
done

for version in 20.04 22.04 ; do
    docker build nest -t pyrocko-nest-ubuntu-$version --build-arg base_image=ubuntu:$version
    docker build fat-nest -t pyrocko-fat-nest-ubuntu-$version --build-arg base_image=pyrocko-nest-ubuntu-$version
    docker build build-deb -t pyrocko-build-deb-ubuntu-$version --build-arg base_image=ubuntu:$version
    docker build test-deb -t pyrocko-test-deb-ubuntu-$version --build-arg base_image=ubuntu:$version
done
