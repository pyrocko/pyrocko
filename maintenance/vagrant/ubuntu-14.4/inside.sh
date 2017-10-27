#!/bin/bash

set -e

branch=$1
if [ -z "$branch" ]; then
    branch=master
fi

pyrockodir="pyrocko-$branch"
outfile_py3="/vagrant/test-$branch.py3.out"
outfile_py2="/vagrant/test-$branch.py2.out"

cd $HOME
sudo apt-get update -y
sudo apt-get install -y git python-setuptools python3-setuptools

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b $branch https://github.com/pyrocko/pyrocko.git "$pyrockodir"
cd "$pyrockodir"
ln -s /pyrocko-test-data test/data

python3 setup.py install_prerequisites --force-yes
sudo python3 setup.py install -f
rm -f "$outfile_py3"
nosetests3 test > >(tee -a "$outfile_py3") 2> >(tee -a "$outfile_py3" >&2) || /bin/true

python setup.py install_prerequisites --force-yes
sudo python setup.py install -f
rm -f "$outfile_py2"
nosetests test > >(tee -a "$outfile_py2") 2> >(tee -a "$outfile_py2" >&2) || /bin/true
