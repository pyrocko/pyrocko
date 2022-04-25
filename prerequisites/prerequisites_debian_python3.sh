#!/bin/sh

sudo apt-get update -y
sudo apt-get install -y make git python3-dev python3-setuptools python3-pip python3-wheel
sudo apt-get install -y python3-numpy python3-numpy-dev python3-scipy python3-matplotlib
sudo apt-get install -y python3-pyqt5 python3-pyqt5.qtopengl python3-pyqt5.qtsvg
sudo apt-get install -y python3-pyqt5.qtwebengine || sudo apt-get install -y python3-pyqt5.qtwebkit
sudo apt-get install -y python3-yaml python3-progressbar python3-jinja2
sudo apt-get install -y python3-requests
sudo apt-get install -y python3-coverage python3-nose
