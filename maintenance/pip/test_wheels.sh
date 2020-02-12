#!/bin/bash

set -e

cd /src

pip install -r requirements.txt jinja2 pybtex
pip install jinja2 pybtex
pip install -f /wheels --no-index pyrocko
python -m nose test.test_util
