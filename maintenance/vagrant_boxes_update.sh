#!/bin/bash

set -e

rm -f vagrant/*/*.out

for box in `ls vagrant` ; do 
    cd "vagrant/$box"
    vagrant box update
    cd ../..
done
