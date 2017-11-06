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
outfile_py2="/vagrant/test-$branch.py2.out"

rm -f "$outfile_py2"

cd $HOME
sudo yum -y install git python-setuptools

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b "$branch" "/vagrant/pyrocko.git" "$pyrockodir"
cd "$pyrockodir"
ln -s "/vagrant/pyrocko-test-data" "test/data"

python setup.py install_prerequisites --force-yes && \
    sudo python setup.py install -f && \
    python --version >> "$outfile_py2" && \
    python -m pyrocko.print_version >> "$outfile_py2" && \
    nosetests "$thetest" > >(tee -a "$outfile_py2") 2> >(tee -a "$outfile_py2" >&2) || \
    /bin/true
