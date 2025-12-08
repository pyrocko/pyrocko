#!/bin/bash

set -e

if [ ! -f "build_packages_windows.sh" ] ; then
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
    if [ -z "$ANACONDA_API_TOKEN" ] ; then
        echo "need anaconda api token as env variable"
        exit 1
    fi
fi

conda install -q -y conda-build anaconda-client numpy

if [ "$ACTION" == "upload" ] ; then
    conda config --set anaconda_upload yes
else
    conda config --set anaconda_upload no
fi

conda-build --python 3.10 build_windows
conda-build --python 3.11 build_windows
conda-build --python 3.12 build_windows
conda-build --python 3.13 build_windows
conda-build --python 3.14 build_windows
