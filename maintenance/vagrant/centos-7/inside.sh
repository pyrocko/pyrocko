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
outfile_py2="/vagrant/test-$branch.py2.out"

rm -f "$outfile_py3"
rm -f "$outfile_py2"

cd $HOME
sudo yum -y install git make gcc mesa-libGL

if [ ! -f "anaconda3.sh" ] ; then
    curl 'https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh' -o anaconda3.sh
fi

if [ ! -d "anaconda3" ] ; then
    sh anaconda3.sh -u -b
fi

export PATH="/home/vagrant/anaconda3/bin:$PATH"

conda install -y -q -c pyrocko pyrocko

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b "$branch" "/vagrant/pyrocko.git" "$pyrockodir"
cd "$pyrockodir"
ln -s "/vagrant/pyrocko-test-data" "test/data"
ln -s "/vagrant/example_run_dir" "test/example_run_dir"

python setup.py install -f && \
    python -m pyrocko.print_version deps >> "$outfile_py3" && \
    python -m nose "$thetest" > >(tee -a "$outfile_py3") 2> >(tee -a "$outfile_py3" >&2) || \
    /bin/true

python2=/usr/bin/python2
sudo "$python2" install_prerequisites.py --yes && \
    sudo "$python2" setup.py install -f && \
    "$python2" -m pyrocko.print_version deps >> "$outfile_py2" && \
    "$python2" -m nose "$thetest" > >(tee -a "$outfile_py2") 2> >(tee -a "$outfile_py2" >&2) || \
    /bin/true
