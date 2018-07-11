#!/bin/bash

set -e

rm -f vagrant/*/*.out

for box in `ls vagrant` ; do 
    cd "vagrant/$box"
    vagrant destroy -f
    cd ../..
done
