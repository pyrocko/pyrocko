#!/bin/bash

set -e

pip3 install -r requirements-all.txt
pip3 install -f wheels --no-index pyrocko
python3 -m nose test.base.test_util test.base.test_eikonal
