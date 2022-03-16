#!/bin/bash

destination=$1

if [ -z "$destination" ] ; then
    echo "usage: upload_wheels.sh (testing|live)"
fi

[ -d wheels ] || mkdir wheels
[ -d dist ] || mkdir dist

payload=`find wheels dist -name 'pyrocko-*'`

if [ "$destination" == 'live' ] ; then
    twine upload "$payload" \
        --username="$PYPI_USERNAME" --password="$PYPI_PASSWORD" \
        --skip-existing --disable-progress-bar --comment='*grunz-grunz*'
else
    twine upload --repository-url https://test.pypi.org/legacy/ \
        "$payload" \
        --username="$PYPI_USERNAME" --password="$PYPI_PASSWORD" \
        --skip-existing --disable-progress-bar --comment='*grunz-grunz*'
fi
