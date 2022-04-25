#!/bin/bash

set -e

branch="$1"
if [ -z "$branch" ]; then
    branch=master
fi

thetest="$2"
if [ -z "$thetest" ]; then
    thetest="test"
fi

pyrockodir="pyrocko-$branch"
outfile_py3="/vagrant/test-$branch.py3.out"

rm -f "$outfile_py3"

cd $HOME
/vagrant/wait_dpkg_locks.sh

sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y git python3-setuptools xvfb

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b $branch "/vagrant/pyrocko.git" "$pyrockodir"
cd "$pyrockodir"
ln -s "/pyrocko-test-data" "test/data"
ln -s "/vagrant/example_run_dir" "test/example_run_dir"

sudo python3 install_prerequisites.py --yes && \
    sudo pip3 install --no-deps --force-reinstall --no-build-isolation . && \
    python3 -m pyrocko.print_version deps >> "$outfile_py3" && \
    xvfb-run -s '-screen 0 640x480x24' python3 -m nose "$thetest" > >(tee -a "$outfile_py3") 2> >(tee -a "$outfile_py3" >&2) || \
    /bin/true
