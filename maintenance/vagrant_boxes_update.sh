#!/bin/bash

set -e

rm -f vagrant/*/*.out

for box in `ls vagrant` ; do 
    cd "vagrant/$box"
    vagrant plugin repair
    vagrant plugin expunge --reinstall --force
    vagrant box update
    cd ../..
done
