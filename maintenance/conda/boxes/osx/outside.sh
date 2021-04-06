#!/bin/bash

set -e

#running=`vagrant global-status | grep running | grep conda/boxes/osx` || /bin/true
#
#if [ ! -z "$running" ]; then
#    echo "vagrant box already running:" $running
#    exit 1
#fi

if [ ! -z "$(git status --untracked-files=no --porcelain)" ]; then
    echo "repos not clean"
    exit 1
fi

if [ -d "pyrocko.git" ]; then
    rm -rf "pyrocko.git"
fi

branch=`git rev-parse --abbrev-ref HEAD`

action="$1"

if [ -z "$action" ] ; then
    echo "usage: outside.sh (dryrun|upload)"
    exit 1
fi

git clone --bare "../../../.." "pyrocko.git"

cat >env.sh <<EOF
export CONDA_USERNAME=$CONDA_USERNAME
export CONDA_PASSWORD=$CONDA_PASSWORD
EOF

vagrant up

set +e
vagrant ssh -- -t /vagrant/inside.sh "$branch" "$action"
STATE=$?
set -e

rm env.sh

vagrant halt
vagrant destroy -f
exit $STATE
