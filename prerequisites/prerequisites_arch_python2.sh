#!/bin/bash

sudo pacman -Syu --noconfirm --needed git make gcc python2 python2-setuptools \
    python2-numpy python2-scipy python2-matplotlib \
    python2-pyqt5 qt5-webengine qt5-svg qt5-webkit \
    python2-cairo python2-opengl \
    python2-requests python2-yaml python2-jinja python2-future \
    python2-nose python2-coverage
