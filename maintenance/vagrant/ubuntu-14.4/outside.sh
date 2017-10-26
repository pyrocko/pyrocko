#!/bin/bash

set -e

branch=$1

if [ -z "$branch" ]; then
    branch=master
fi

echo $branch

vagrant up
vagrant ssh -- -X /vagrant/inside.sh $branch
vagrant halt
