#!/bin/bash

set -e

vagrant up
vagrant ssh -- -t /vagrant/inside.sh
vagrant halt
