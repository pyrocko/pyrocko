#!/bin/bash

set -e

if [ ! -f deploy-pypi.sh ] ; then
    echo "must be run from pyrocko's maintenance directory"
    exit 1
fi

branch="$1"
if [ -z "$branch" ]; then
    branch=`git rev-parse --abbrev-ref HEAD`
fi

if [ ! -e pypirc ] ; then
    echo "!! NO pypirc found !!
Create a 'pypirc' with the following content:

'''
[distutils]
index-servers=
  pypi
  testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = pyrocko
password = <password>

[pypi]
repository = https://upload.pypi.org/legacy/
username = pyrocko
password = <password>
'''
    "
    exit 1
fi

cd pip-boxes/centos-7

vagrant halt
echo "Building Pip packages on $box"
./outside.sh $branch
