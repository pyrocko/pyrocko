#!/bin/bash
# To return a failure if any commands inside fail
set -e

MINICONDA_URL="http://repo.continuum.io/miniconda"
CONDA_PREFIX=$HOME/miniconda3

BUILD_DIR=`dirname $0`/pyrocko-anaconda-deploy

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
    MINICONDA_FILE=Miniconda3-latest-MacOSX-x86_64.sh
else
    MINICONDA_FILE=Miniconda3-latest-Linux-x86_64.sh
fi

wget $MINICONDA_URL/$MINICONDA_FILE -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -u
export PATH=$PATH:$CONDA_PREFIX/bin

conda install -y conda-build anaconda-client
anaconda logout
anaconda login --username $ANACONDA_USER --password $ANACONDA_PASSWORD

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cp anaconda-meta.yaml $BUILD_DIR/meta.yaml
wget http://pyrocko.org/v0.3/_images/pyrocko_shadow.png -O $BUILD_DIR/snuffler.png

cd $BUILD_DIR
conda config --set anaconda_upload yes
conda-build .
cd ..
rm -rf $BUILD_DIR

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
