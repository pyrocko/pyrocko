#!/bin/bash
set -e
VERSION=`python -c "import pyrocko; print(pyrocko.__version__);"`

if [ ! -f maintenance/update-docs.sh ] ; then
    echo "must be run from pyrocko's toplevel directory"
    exit 1
fi

cd doc
make clean; make html $1
mv build/html build/$VERSION

read -r -p "Are your sure to update live docs at http://pyrocko.org/docs/ [y/N]?" resp
case $resp in
    [yY][eE][sS]|[yY] )
        scp -r build/$VERSION pyrocko@hive:/var/www/pyrocko.org/docs;
        echo "Linking docs/$VERSION to docs/current";
        ssh pyrocko@hive "rm -rf /var/www/pyrocko.org/docs/current; ln -s /var/www/pyrocko.org/docs/$VERSION /var/www/pyrocko.org/docs/current";
        ;;
    * ) ;;
esac

cd ..
