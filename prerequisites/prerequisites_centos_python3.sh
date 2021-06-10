#!/bin/bash

sudo yum -y install epel-release dnf-plugins-core
sudo yum config-manager --set-enabled powertools

sudo yum -y install make gcc git python3 python3-yaml python3-matplotlib python3-numpy \
    python3-scipy python3-requests python3-coverage python3-nose python3-jinja2 \
    python3-qt5 python3-matplotlib-qt5
