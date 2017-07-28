#!/bin/bash
# To return a failure if any commands inside fail
set -e

MINICONDA_URL="http://repo.continuum.io/miniconda"
CONDA_PREFIX=$HOME/miniconda3

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
    MINICONDA_FILE=Miniconda3-latest-MacOSX-x86_64.sh
else
    MINICONDA_FILE=Miniconda3-latest-Linux-x86_64.sh
fi

wget $MINICONDA_URL/$MINICONDA_FILE -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=$PATH:$CONDA_PREFIX/bin

conda install -y conda-build anaconda-client
anaconda-client login --username $ANACONDA_USER --password $ANACONDA_PASSWORD

mkdir pyrocko
cp anaconda-meta.yaml pyrocko/meta.yaml
wget http://pyrocko.org/v0.3/_images/pyrocko_shadow.png -O pyrocko/snuffler.png

cd pyrocko
conda config --set anaconda_upload yes
conda-build .
rm -rf pyrocko

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
