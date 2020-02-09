#!/bin/bash

sudo pacman -Syu --noconfirm --needed git make gcc python python-setuptools \
    python-numpy python-scipy python-matplotlib \
    python-pyqt5 qt5-webengine qt5-svg qt5-webkit \
    python-cairo python-opengl python-progressbar \
    python-requests python-yaml python-jinja python-future \
    python-nose python-coverage
