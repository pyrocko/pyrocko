#!/bin/bash

set -e

branch=$1
if [ -z "$branch" ]; then
    branch=master
fi

pyrockodir="pyrocko-$branch"
outfile="/vagrant/test-$branch.out"

cd $HOME
sudo yum -y install git python-setuptools python34-setuptools

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b $branch https://github.com/pyrocko/pyrocko.git "$pyrockodir"
cd "$pyrockodir"
ln -s /vagrant/pyrocko-test-data test/data
python setup.py install_prerequisites --force-yes
sudo python setup.py install -f
rm -f "$outfile"
nosetests test > >(tee -a "$outfile") 2> >(tee -a "$outfile" >&2)
