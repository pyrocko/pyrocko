#!/bin/bash

set -e

branch="$1"
if [ -z "$branch" ]; then
    branch=master
fi

thetest="$2"
if [ -z "$thetest" ]; then
    thetest="test"
fi

pyrockodir="pyrocko-$branch"
outfile_py3="/vagrant/test-$branch.py3.out"

rm -f "$outfile_py3"

cd $HOME
sudo pacman -Syu --noconfirm --needed git python-setuptools \
    xorg-server-xvfb xorg-fonts-100dpi xorg-fonts-75dpi \
    xorg-fonts-misc patch

if [ -e "$pyrockodir" ] ; then
    sudo rm -rf "$pyrockodir"
fi
git clone -b $branch "/vagrant/pyrocko.git" "$pyrockodir"
cd "$pyrockodir"
ln -s "/pyrocko-test-data" "test/data"
ln -s "/vagrant/example_run_dir" "test/example_run_dir"

mkdir -p "$HOME/.config/matplotlib"
echo "backend : agg" > "$HOME/.config/matplotlib/matplotlibrc"

python3 install.py deps system --yes && \
    python3 install.py system --yes && \
    python3 -m pyrocko.print_version deps >> "$outfile_py3" && \
    xvfb-run -s '-screen 0 640x480x24' python3 -m pytest -v  "$thetest" > >(tee -a "$outfile_py3") 2> >(tee -a "$outfile_py3" >&2) || \
    /bin/true
