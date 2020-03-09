#!/bin/sh

sudo apt-get update -y
sudo apt-get install -y make git python-dev python-setuptools
sudo apt-get install -y python-numpy python-numpy-dev python-scipy python-matplotlib
sudo apt-get install -y python-qt4 python-qt4-gl
sudo apt-get install -y python-pyqt5 python-pyqt5.qtopengl python-pyqt5.qtsvg
sudo apt-get install -y python-pyqt5.qtwebengine || sudo apt-get install -y python-pyqt5.qtwebkit
sudo apt-get install -y python-yaml python-progressbar python-jinja2
sudo apt-get install -y python-requests
sudo apt-get install -y python-coverage python-nose
