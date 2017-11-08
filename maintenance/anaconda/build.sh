$PYTHON setup.py install

if [ `uname` == Darwin ]
then
    cp $RECIPE_DIR/snuffler_mac.command $PREFIX/bin
fi
