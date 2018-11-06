#!/bin/bash
# To return a failure if any commands inside fail
# More: https://conda.io/docs/user-guide/tutorials/build-pkgs.html
set -e
MINICONDA_URL="https://repo.continuum.io/miniconda"
CONDA_PREFIX="$HOME/miniconda3"
BUILD_DIR=`dirname $0`
export PATH=$PATH:$CONDA_PREFIX/bin

if [ `uname` == "Darwin" ]; then
    MINICONDA_FILE="Miniconda3-latest-MacOSX-x86_64.sh"
else
    MINICONDA_FILE="Miniconda3-latest-Linux-x86_64.sh"
fi

# Install Miniconda
read -r -p "Do you want to download and install $MINICONDA_FILE [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        curl "$MINICONDA_URL/$MINICONDA_FILE" -o "$BUILD_DIR/miniconda.sh";
        chmod +x "$BUILD_DIR/miniconda.sh"
        ./$BUILD_DIR/miniconda.sh -b -u
        conda install -y conda-build anaconda-client
        ;;
    *)
        ;;
esac

# Install Anaconda client and build tools
read -r -p "Do you want to upload pyrocko to Anaconda (https://anaconda.org/pyrocko/pyrocko) [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        anaconda logout;
        anaconda login --username pyrocko;
        conda config --set anaconda_upload yes;
        ;;
    * ) conda config --set anaconda_upload no;
        ;;
esac
conda-build --python 3.6 $BUILD_DIR
conda-build --python 3.7 $BUILD_DIR

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
