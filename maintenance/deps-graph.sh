#!/bin/sh

cat >clusters <<EOF
pyrocko/snufflings
pyrocko/fomosto
pyrocko/fomosto_report
pyrocko/gf
pyrocko/topo
EOF

sfood --ignore=pyrocko/util.py --ignore-unused --internal  /usr/local/lib/python2.7/dist-packages/pyrocko  apps/snuffler apps/automap apps/fomosto apps/cake apps/jackseis | grep -v /util.py | grep -v /guts.py | grep -v /io_common.py | sfood-cluster -f clusters > deps

sfood-graph -p deps2 > deps2.dot

dot -Tpdf -Gmargin=0.1 -Grankdir=TB -Gratio=0.2 deps2.dot -o deps2.pdf

