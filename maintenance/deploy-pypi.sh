#!/bin/bash
if [ ! -f deploy-pypi.sh ] ; then
    echo "must be run from pyrocko's maintenance directory"
    exit 1
fi

branch="$1"
if [ -z "$branch" ]; then
    branch=master
fi

echo "Starting Pyrocko PyPi Deployment Branch $branch"

if [ ! -e ~/.pypirc ] ; then
    echo "!! NO ~/.pypirc found !!
Create a ~/.pypirc with the following content:

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
    exit
fi

read -r -p "Do you want to deploy local branch $branch on https://pypi.python.org [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        git clone -b $branch "../" "pip-pyrocko.git"
        cd "pip-pyrocko.git"
        python setup.py sdist upload -r pypi
        cd -
        ;;
    * ) ;;
esac
