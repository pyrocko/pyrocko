#!/bin/bash
set -e

eval "$(/usr/local/bin/brew shellenv)"
rm -rf venv_x86_64
python3 -m venv venv_x86_64 --system-site-packages
source venv_x86_64/bin/activate
pip install .
python setup.py py2app
mv dist/Snuffler.app dist/Snuffler_x86_64.app
cd dist
zip -r Snuffler_x86_64.app.zip Snuffler_x86_64.app
rm -rf Snuffler_x86_64.app
