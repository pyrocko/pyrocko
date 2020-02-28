FROM debian:stable

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y xvfb libgles2-mesa \
        make git python-dev python-setuptools \
        python-numpy python-numpy-dev python-scipy python-matplotlib \
        python-qt4 python-qt4-gl \
        python-pyqt5 python-pyqt5.qtopengl python-pyqt5.qtsvg \
        python-yaml python-progressbar python-jinja2 \
        python-requests \
        python-coverage python-nose \
        python3-dev python3-setuptools \
        python3-numpy python3-numpy-dev python3-scipy python3-matplotlib \
        python3-pyqt4 python3-pyqt4.qtopengl \
        python3-pyqt5 python3-pyqt5.qtopengl python3-pyqt5.qtsvg \
        python3-yaml python3-progressbar python3-jinja2 \
        python3-requests \
        python3-coverage python3-nose ; \
    apt-get install -y python-pyqt5.qtwebengine || apt-get install -y python-pyqt5.qtwebkit ; \
    apt-get install -y python-future || easy_install future ; \
    apt-get install -y python3-pyqt5.qtwebengine || apt-get install -y python3-pyqt5.qtwebkit ; \
    apt-get install -y python3-future || easy_install3 future ; \
    apt-get install -y python-pip python3-pip
RUN pip3 install flake8


