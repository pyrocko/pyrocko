#!/bin/bash

set -e

pyrockodir="pyrocko"

cd $HOME

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi

git clone "https://github.com/pyrocko/pyrocko.git" "$pyrockodir"
cd "$pyrockodir/maintenance"

anaconda/deploy_anaconda3.sh
anaconda/deploy_anaconda2.sh
