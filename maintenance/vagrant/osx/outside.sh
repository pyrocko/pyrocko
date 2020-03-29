#!/bin/bash

set -e

branch="$1"

if [ -z "$branch" ]; then
    branch=`git rev-parse --abbrev-ref HEAD`
fi

thetest="$2"

if [ -z "$thetest" ]; then
    thetest="test"
fi

if [ -d "pyrocko.git" ]; then
    rm -rf "pyrocko.git"
fi
git clone --bare "../../.." "pyrocko.git"
cp -r "../../../test/example_run_dir" "."

echo "testing branch: $branch"
echo "running test: $thetest"
rm -f log.out
echo "testing branch $branch" >> log.out
date -uIseconds >> log.out
vagrant up
vagrant rsync
vagrant ssh -- /vagrant/inside.sh "$branch" "$thetest" > >(tee -a "log.out") 2> >(tee -a "log.out" >&2) || /bin/true
vagrant rsync-back || ( vagrant plugin install vagrant-rsync-back && vagrant rsync-back )
vagrant halt
date -uIseconds >> log.out
