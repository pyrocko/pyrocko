#!/bin/bash

set -e

if [ ! -f "build_packages_osx_arm64.sh" ] ; then
    echo 'must be run from inside maintenance/conda'
    exit 1
fi

if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "repos not clean"
    exit 1
fi


ACTION="$1"

if [ -z "$ACTION" ] ; then
    echo "usage: build_packages.sh (dryrun|upload)"
    exit 1
fi

if [ "$ACTION" == "upload" ] ; then
    if [ -z "$CONDA_USERNAME" -o -z "$CONDA_PASSWORD" ] ; then
        echo "need anaconda credentials as env variables"
        exit 1
    fi
fi

conda install -q -y conda-build anaconda-client numpy

if [ "$ACTION" == "upload" ] ; then
    conda config --set anaconda_upload yes
    function anaconda_logout {
        anaconda logout
    }
    trap anaconda_logout EXIT
else
    conda config --set anaconda_upload no
fi

conda-build --python 3.10 build
conda-build --python 3.11 build
conda-build --python 3.12 build
conda-build --python 3.13 build
conda-build --python 3.14 build

if [ "$ACTION" == "upload" ] ; then
    trap - EXIT
    anaconda_logout
fi
