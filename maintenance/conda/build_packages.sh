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

ORIGPATH="$PATH"

CONDA_URL="https://repo.anaconda.com/miniconda"
CONDA_PREFIX="/tmp/miniconda3"
CONDA_INSTALLER="miniconda3.sh"

export PATH="$CONDA_PREFIX/bin:$ORIGPATH"

CONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"

# Install Miniconda

HERE=`pwd`
cd "$HOME"

if [ ! -f "$CONDA_INSTALLER" ] ; then
    echo "getting conda from:" "$CONDA_URL/$CONDA_FILE"
    curl "$CONDA_URL/$CONDA_FILE" -o "$CONDA_INSTALLER"
    chmod +x "$CONDA_INSTALLER"
    rm -rf "$CONDA_PREFIX"
fi

if [ ! -d "$CONDA_PREFIX" ] ; then
    "./$CONDA_INSTALLER" -b -u -p "$CONDA_PREFIX"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    conda install -q -y conda-build anaconda-client numpy
fi

cd "$HERE"

if [ "$ACTION" == "upload" ] ; then
    conda config --set anaconda_upload yes
else
    conda config --set anaconda_upload no
fi

conda-build --python 3.10 build
conda-build --python 3.11 build
conda-build --python 3.12 build
conda-build --python 3.13 build
conda-build --python 3.14 build
