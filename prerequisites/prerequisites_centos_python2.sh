#!/bin/bash

sudo yum -y install make gcc git python python-yaml python-matplotlib numpy \
    scipy python-requests python-coverage python-nose python-jinja2

sudo yum -y install python-future || sudo easy_install future
