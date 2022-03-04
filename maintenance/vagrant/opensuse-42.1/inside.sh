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
sudo zypper -n refresh
sudo zypper -n update
sudo zypper -n install git python3-setuptools

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b $branch "/vagrant/pyrocko.git" "$pyrockodir"
cd "$pyrockodir"
ln -s "/pyrocko-test-data" "test/data"
ln -s "/vagrant/example_run_dir" "test/example_run_dir"

sudo python3 install_prerequisites.py --yes && \
    sudo python3 setup.py install -f && \
    python3 -m pyrocko.print_version deps >> "$outfile_py3" && \
    python3 -m nose "$thetest" > >(tee -a "$outfile_py3") 2> >(tee -a "$outfile_py3" >&2) || \
    /bin/true
