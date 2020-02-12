#!/bin/bash

set -e

cd /src

pip install -r requirements-all.txt
pip install -f /wheels --no-index pyrocko
python -m nose test.test_util:UtilTestCase.testTime
