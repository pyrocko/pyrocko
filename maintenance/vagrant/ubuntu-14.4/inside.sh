#!/bin/bash

set -e

branch=$1
if [ -z "$branch" ]; then
    branch=master
fi

pyrockodir="pyrocko-$branch"
outfile="/vagrant/test-$branch.out"

cd $HOME
sudo apt-get install -y git python-setuptools python3-setuptools

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b $branch https://github.com/pyrocko/pyrocko.git "$pyrockodir"
cd "$pyrockodir"
ln -s /pyrocko-test-data test/data
python3 setup.py install_prerequisites --force-yes
sudo python3 setup.py install -f
rm -f "$outfile"
nosetests3 test > >(tee -a "$outfile") 2> >(tee -a "$outfile" >&2)
