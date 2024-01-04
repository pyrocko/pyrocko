#!/bin/bash

set -e

cd $HOME

sudo echo 'preparing miniconda for arm64 and x86_64 for conda build on osx'

if [ ! -d prepare_miniconda_packages ] ; then
    mkdir prepare_miniconda_packages
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o prepare_miniconda_packages/miniconda3_arm64.sh
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o prepare_miniconda_packages/miniconda3_x86_64.sh
    curl -L https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX10.9.sdk.tar.xz -o prepare_miniconda_packages/MacOSX10.9.sdk.tar.xz
    curl -L https://github.com/phracker/MacOSX-SDKs/releases/download/11.3/MacOSX11.0.sdk.tar.xz -o prepare_miniconda_packages/MacOSX11.0.sdk.tar.xz
fi

rm -rf miniconda3_arm64
mkdir -p miniconda3_arm64
bash prepare_miniconda_packages/miniconda3_arm64.sh -b -u -p ~/miniconda3_arm64
source miniconda3_arm64/bin/activate
conda update -y -n base -c defaults conda
conda deactivate

rm -rf miniconda3_x86_64
mkdir -p miniconda3_x86_64
bash prepare_miniconda_packages/miniconda3_x86_64.sh -b -u -p ~/miniconda3_x86_64
source miniconda3_x86_64/bin/activate
conda update -y -n base -c defaults conda
conda deactivate

# mix of conda clang and new sdks leads to errors; use old sdk version
sudo tar xf prepare_miniconda_packages/MacOSX10.9.sdk.tar.xz -C /opt
sudo tar xf prepare_miniconda_packages/MacOSX11.0.sdk.tar.xz -C /opt

cat > .condarc <<EOF
anaconda_upload: false
conda_build:
  config_file: ~/.conda/conda_build_config.yaml
EOF

mkdir -p .conda
cat > .conda/conda_build_config.yaml <<EOF
CONDA_BUILD_SYSROOT:
  - /opt/MacOSX10.9.sdk      # [osx and x86_64]
  - /opt/MacOSX11.0.sdk      # [osx and arm64]
EOF

