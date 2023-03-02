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

rm -rf pyrocko.git pyrocko-test-data example_run_dir *.out *.log
git clone --bare "../../.." "pyrocko.git"
cp -a "../../../test/data" pyrocko-test-data
cp -a "../../../test/example_run_dir" example_run_dir

echo "testing branch: $branch"
echo "running test: $thetest"
echo "testing branch $branch" >> log.out
date -uIseconds >> log.out
vagrant up
vagrant rsync
vagrant ssh -- /vagrant/inside.sh "$branch" "$thetest" > >(tee -a "log.out") 2> >(tee -a "log.out" >&2) || /bin/true
vagrant rsync-back || ( vagrant plugin install vagrant-rsync-back && vagrant rsync-back )
vagrant halt
date -uIseconds >> log.out
