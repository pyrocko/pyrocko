#!/bin/bash

set -e

rm -f vagrant/*/*.out

branch=$1

if [ -z "$branch" ]; then
    echo 'usage: vagrant_test.sh <branch>'
    exit 1
fi

for box in `ls vagrant` ; do 
    cd "vagrant/$box"
    vagrant halt
    cd ../..
done

for box in `ls vagrant` ; do 
    echo "testing box $box"
    cd "vagrant/$box"
    ./outside.sh "$branch"
    cd ../..
done
