#!/bin/bash
if [ ! -f deploy-anaconda_macvm.sh ] ; then
    echo "must be run from pyrocko's maintenance directory"
    exit 1
fi


branch="$1"
if [ -z "$branch" ]; then
    branch=master
fi

echo "Building pyrocko for Anaconda on branch $branch"
rm -rf "anaconda/pyrocko.git"
git clone -b $branch "../" "anaconda/pyrocko.git"
rm -rf "anaconda/pyrocko.git/maintenance/anaconda/meta.yaml"
# rm -rf "anaconda/pyrocko.git/.git"

cd anaconda/
vagrant up
cd -
