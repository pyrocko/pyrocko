#!/bin/sh

set -e

if [ ! -f libmseed/libmseed.a ]; then
    rm -rf libmseed
    tar -xzf libmseed-2.12.tar.gz
    cd libmseed
    make gcc
    cd ..
fi

if [ ! -d evalresp-3.3.0/lib ]; then
    rm -rf evalresp-3.3.0
    tar -xzf evalresp-3.3.0.tar.gz
    cd evalresp-3.3.0
    ./configure --prefix="`pwd`" --libdir="`pwd`/lib" CFLAGS=-fPIC
    make
    make install
    cd ..
fi

