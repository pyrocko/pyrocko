#!/bin/bash

set -e

if [ ! -f "build_packages.sh" ] ; then
    echo 'must be run from inside maintenance/conda'
    exit 1
fi

if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "repos not clean"
    exit 1
fi


BRANCH="$1"
ACTION="$2"

if [ -z "$BRANCH" -o -z "$ACTION" ] ; then
    echo "usage: build_packages.sh <branch> (dryrun|upload)"
    exit 1
fi

if [ "$ACTION" == "UPLOAD" ] ; then
    if [ -z "$CONDA_USERNAME" -o -z "$CONDA_PASSWORD" ] ; then
        echo "need anaconda credentials as env variables"
        exit 1
    fi
fi

ORIGPATH="$PATH"

for VERSION in 3 2 ; do

    CONDA_URL="https://repo.continuum.io/miniconda"
    HERE=`pwd`
    CONDA_PREFIX="$HERE/miniconda${VERSION}"
    CONDA_INSTALLER="miniconda${VERSION}.sh"

    export PATH="$CONDA_PREFIX/bin:$ORIGPATH"

    if [ `uname` == "Darwin" ]; then
        CONDA_FILE="Miniconda${VERSION}-latest-MacOSX-x86_64.sh"
    else
        CONDA_FILE="Miniconda${VERSION}-latest-Linux-x86_64.sh"
    fi

    # Install Miniconda

    if [ ! -f "$CONDA_INSTALLER" ] ; then
        curl "$CONDA_URL/$CONDA_FILE" -o "$CONDA_INSTALLER";
        chmod +x "$CONDA_INSTALLER"
        rm -rf "$CONDA_PREFIX"
    fi

    if [ ! -d "$CONDA_PREFIX" ] ; then
        "./$CONDA_INSTALLER" -b -u -p "$CONDA_PREFIX"
        conda install -y conda-build conda-verify anaconda-client
    fi

    if [ -d "build/pyrocko.git" ] ; then
        rm -rf "build/pyrocko.git"
    fi

    git clone -b $BRANCH "../.." "build/pyrocko.git"
    rm -rf build/pyrocko.git/.git
    rm -rf build/pyrocko.git/maintenance/conda

    if [ "$ACTION" == "upload" ] ; then
        anaconda login --username "$CONDA_USERNAME" --password "$CONDA_PASSWORD" --hostname conda-builder-`uname`
        conda config --set anaconda_upload yes
        function anaconda_logout {
            anaconda logout
        }
        trap anaconda_logout EXIT
    else
        conda config --set anaconda_upload no
    fi

    if [ "$VERSION" == "3" ] ; then
        conda-build --python 3.6 build
        conda-build --python 3.7 build
        conda-build --python 3.8 build
    fi

    if [ "$VERSION" == "2" ] ; then
        conda-build build
    fi

    if [ "$ACTION" == "upload" ] ; then
        trap - EXIT
        anaconda_logout
    fi
done
