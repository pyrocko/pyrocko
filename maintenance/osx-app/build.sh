#!/bin/bash

set -e

eval "$(/opt/homebrew/bin/brew shellenv)"
rm -rf venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install .
python setup.py py2app
mv dist/Snuffler.app dist/Snuffler_arm64.app
cd dist
zip -r Snuffler_arm64.app.zip Snuffler_arm64.app
rm -rf Snuffler_arm64.app
