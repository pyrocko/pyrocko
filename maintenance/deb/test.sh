#!/bin/bash

set -e

revision=$1
debianversion=$2

if [ -z "$revision" ] || [ -z "$debianversion" ]; then
    echo "usage: test.sh <revision> <debianversion>"
    exit 1
fi

apt install -q -y ./deb-packages-${debianversion}/pyrocko_*.*.*-${revision}_*.deb >/dev/null 2>/dev/null
mkdir -p /pyrocko-test-data/example_run_dir
ln -sf /pyrocko-test-data/gf_stores/* /pyrocko-test-data/example_run_dir
PYROCKO_TEST_DATA=/pyrocko-test-data xvfb-run -s '-screen 0 640x480x24' python3 -m pytest ${@:3}
