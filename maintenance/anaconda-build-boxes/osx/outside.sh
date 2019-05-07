#!/bin/bash

set -e

vagrant up
vagrant ssh -- -t /Users/vagrant/anaconda/inside.sh
vagrant halt
