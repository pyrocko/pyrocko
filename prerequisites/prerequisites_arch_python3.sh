#!/bin/bash

sudo pacman -Syu --noconfirm --needed git make gcc python python-setuptools \
    python-numpy python-scipy python-matplotlib \
    python-cairo python-pyqt4 python-opengl python-progressbar \
    python-requests python-yaml python-jinja python-future \
    python-nose python-coverage
