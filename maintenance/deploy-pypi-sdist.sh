#!/bin/bash

set -e

if [ ! -f deploy-pypi-sdist.sh ] ; then
    echo "must be run from pyrocko's maintenance directory"
    exit 1
fi

branch="$1"
if [ -z "$branch" ]; then
    branch=`git rev-parse --abbrev-ref HEAD`
fi

pyrockodir='pyrocko-pip'

if [ -e "$pyrockodir" ] ; then
    rm -rf "$pyrockodir"
fi

git clone -b $branch .. "$pyrockodir"

cd "$pyrockodir"

python setup.py sdist
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

