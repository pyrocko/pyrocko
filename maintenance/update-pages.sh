#!/bin/bash
if [ ! -f maintenance/update-pages.sh ] ; then
    echo "must be run from pyrocko's toplevel directory"
    exit 1
fi

if [ ! -d pages ] ; then
    git clone -b gh-pages git@github.com:emolch/pyrocko.git pages || exit 1
fi
cd pages || exit 1
git pull origin gh-pages || exit 1
cd ..
cd doc || exit 1
make html || exit 1
cd ..
cp -R doc/_build/html/* pages/ || exit 1
cd pages || exit 1
git add -A || exit 1
git commit || exit 1
git push origin gh-pages

