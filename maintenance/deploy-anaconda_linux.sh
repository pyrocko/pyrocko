#!/bin/bash
# To return a failure if any commands inside fail
# More: https://conda.io/docs/user-guide/tutorials/build-pkgs.html
set -e

MINICONDA_URL="http://repo.continuum.io/miniconda"
CONDA_PREFIX=$HOME/miniconda3

BUILD_DIR=`dirname $0`/anaconda/pyrocko-anaconda-deploy

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
    MINICONDA_FILE=Miniconda3-latest-MacOSX-x86_64.sh
else
    MINICONDA_FILE=Miniconda3-latest-Linux-x86_64.sh
fi

# Install Miniconda
wget $MINICONDA_URL/$MINICONDA_FILE -O anaconda/miniconda.sh
chmod +x anaconda/miniconda.sh
./anaconda/miniconda.sh -b -u
export PATH=$PATH:$CONDA_PREFIX/bin

# Install Anaconda client and build tools
conda install -y conda-build anaconda-client

# Create build env
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR

cp anaconda/meta.yaml $BUILD_DIR/meta.yaml
cp anaconda/build.sh $BUILD_DIR/build.sh

cd $BUILD_DIR
read -r -p "Do you want to upload pyrocko to Anaconda (https://anaconda.org/pyrocko/pyrocko) [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        anaconda logout;
        anaconda login --username $ANACONDA_USER --password $ANACONDA_PASSWORD;
        conda config --set anaconda_upload yes;
        ;;
    * ) conda config --set anaconda_upload no;
        ;;
esac

conda-build .
cd -

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
