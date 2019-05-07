#!/bin/bash
if [ ! -f anaconda/deploy_anaconda2.sh ] ; then
    echo "must be run from pyrocko's maintenance directory"
    exit 1
fi

branch="$1"
if [ -z "$branch" ]; then
    branch=master
fi

echo "Building pyrocko for Anaconda2 on branch $branch"
rm -rf "anaconda/pyrocko.git"
git clone -b $branch "../" "anaconda/pyrocko.git"
rm -rf "anaconda/pyrocko.git/maintenance/anaconda/meta.yaml"

anaconda/build_anaconda2.sh $1
