#!/bin/bash

set -e

branch=$1

if [ -z "$branch" ]; then
    branch=master
fi

echo $branch

if [ ! -d "pyrocko-test-data" ]; then
    cp -R "../../../test/data" pyrocko-test-data
fi

vagrant up
vagrant ssh -- sudo yum -y install xauth
vagrant ssh -- -X /vagrant/inside.sh $branch
vagrant halt
