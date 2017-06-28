sudo port install git
sudo port install python27
sudo port select python python27
sudo port install py27-numpy
sudo port install py27-scipy
sudo port install py27-matplotlib
sudo port install py27-yaml
sudo port install py27-pyqt4
sudo port install py27-setuptools
pip install -I --user  progressbar
pip install --user -I Jinja2
/opt/local/bin/easy_install-2.7 --user pyavl
python setup.py install --user --install-scripts=/Users/daouts/soft/pyrocko/bin 

git clone git://github.com/pyrocko/pyrocko.git pyrocko
git pull origin master
git pull origin docs 
