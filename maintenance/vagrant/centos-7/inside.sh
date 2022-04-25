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
sudo yum -y install git make gcc mesa-libGL patch

if [ ! -f "miniconda3.sh" ] ; then
    curl 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh' -o miniconda3.sh
fi

if [ ! -d "miniconda3" ] ; then
    sh miniconda3.sh -u -b
fi

export PATH="/home/vagrant/miniconda3/bin:$PATH"

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b "$branch" "/vagrant/pyrocko.git" "$pyrockodir"
cd "$pyrockodir"
ln -s "/vagrant/pyrocko-test-data" "test/data"
ln -s "/vagrant/example_run_dir" "test/example_run_dir"

python install.py deps conda --yes && \
    python install.py user --yes && \
    python -m pyrocko.print_version deps >> "$outfile_py3" && \
    python -m nose "$thetest" > >(tee -a "$outfile_py3") 2> >(tee -a "$outfile_py3" >&2) || \
    /bin/true
