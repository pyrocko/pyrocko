#!/bin/bash
if [ ! -f maintenance/update-pages.sh ] ; then
    echo "must be run from pyrocko's toplevel directory"
    exit 1
fi

if [ ! -d pages ] ; then
    git clone -n git@github.com:emolch/pyrocko.git pages || exit 1
    cd pages || exit 1
    git checkout -b gh-pages origin/gh-pages || exit 1
    cd ..
fi
cd pages || exit 1
git pull origin gh-pages || exit 1
cd ..
cd doc || exit 1
make clean || exit 1
make html || exit 1
cd ..

VERSION=v0.3

if [ ! -d pages/$VERSION ] ; then
    mkdir pages/$VERSION || exit 1
fi
cp -R doc/_build/html/* pages/$VERSION/ || exit 1
cd pages/$VERSION || exit 1

git add * || exit 1
git commit || exit 1
git push origin gh-pages

