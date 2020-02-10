#!/bin/bash

set -e

pyrockodir="pyrocko"

cd $HOME
sudo yum -y install git gcc

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi

git clone "https://git.pyrocko.org/pyrocko/pyrocko.git" "$pyrockodir"
cd "$pyrockodir/maintenance"

anaconda/deploy_anaconda3.sh
anaconda/deploy_anaconda2.sh
