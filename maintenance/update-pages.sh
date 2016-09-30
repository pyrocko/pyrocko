#!/bin/bash

set -e

if [ ! -f maintenance/update-pages.sh ] ; then
    echo "must be run from pyrocko's toplevel directory"
    exit 1
fi

if [ ! -d pages ] ; then
    git clone -b gh-pages -n git@github.com:pyrocko/pyrocko.git pages
fi
cd pages
git pull origin gh-pages
cd ..
cd doc
make clean
make html $1
cd ..

VERSION=v0.3

if [ ! -d pages/$VERSION ] ; then
    mkdir pages/$VERSION
fi
cp -R doc/_build/html/* pages/$VERSION/
cd pages/$VERSION

git add *
git commit
git push origin gh-pages

