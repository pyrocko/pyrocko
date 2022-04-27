#!/bin/bash

destination=$1

if [ -z "$destination" ] ; then
    echo "usage: upload_wheels.sh (testing|live)"
fi

if [ "$destination" == 'live' ] ; then
    twine upload dist/pyrocko-* \
        --username="$PYPI_USERNAME" --password="$PYPI_PASSWORD" \
        --skip-existing --disable-progress-bar
else
    twine upload --repository-url https://test.pypi.org/legacy/ dist/pyrocko-* \
        --username="$PYPI_USERNAME" --password="$PYPI_PASSWORD" \
        --skip-existing --disable-progress-bar
fi
