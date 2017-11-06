#!/bin/bash

set -e

rm -f vagrant/*/*.out

branch=$1
thetest=$2

if [ -z "$branch" ]; then
    echo 'usage: vagrant_tests_run.sh <branch> <test>'
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
    ./outside.sh "$branch" "$thetest"
    cd ../..
done
