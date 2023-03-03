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
sudo yum -y update

sudo yum -y install git make gcc mesa-libGL python3 patch

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b "$branch" "/vagrant/pyrocko.git" "$pyrockodir"
cd "$pyrockodir"
ln -s "/vagrant/pyrocko-test-data" "test/data"
ln -s "/vagrant/example_run_dir" "test/example_run_dir"

python3=/usr/bin/python3
pip3=/usr/bin/pip3

# prevent hangs in numpy.linalg when running from a parimap subprocess...
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

"$python3" install.py deps system --yes && \
    sudo "$pip3" install --no-deps --force-reinstall --upgrade . && \
    "$python3" -m pyrocko.print_version deps >> "$outfile_py3" && \
    "$python3" -m pytest -v  "$thetest" > >(tee -a "$outfile_py3") 2> >(tee -a "$outfile_py3" >&2) || \
    /bin/true
