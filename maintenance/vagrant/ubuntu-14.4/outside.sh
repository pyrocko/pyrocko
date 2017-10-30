#!/bin/bash

set -e

branch=$1

if [ -z "$branch" ]; then
    branch=master
fi

echo "testing branch $branch"
rm -f log.out
echo "testing branch $branch" >> log.out
date -uIseconds >> log.out
vagrant up
vagrant ssh -- -X /vagrant/inside.sh $branch > >(tee -a "log.out") 2> >(tee -a "log.out" >&2) || /bin/true
vagrant halt
date -uIseconds >> log.out
