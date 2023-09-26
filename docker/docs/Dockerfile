FROM debian:stable

RUN apt-get update -y
RUN apt-get upgrade -y

# env requirements
RUN apt-get install -y python3-pip

# build requirements
RUN apt-get install -y make git
RUN apt-get install -y python3-dev python3-setuptools python3-numpy-dev

# docs requirements
RUN apt-get install -y \
    texlive-fonts-recommended texlive-latex-extra \
    texlive-latex-recommended python3-sphinx
RUN pip3 install --break-system-packages sphinx-qt-documentation \
    git+https://git.pyrocko.org/pyrocko/sphinx-sleekcat-theme.git

# base runtime requirements
RUN apt-get install -y \
        python3-numpy python3-scipy python3-matplotlib \
        python3-requests python3-future \
        python3-yaml

# gui runtime requirements
RUN apt-get install -y \
        python3-pyqt5 python3-pyqt5.qtopengl python3-pyqt5.qtsvg \
        python3-pyqt5.qtwebengine python3-pyqt5.qtwebkit
RUN apt-get install -y python3-vtk9 \
    || apt-get install -y python3-vtk8 \
    || apt-get install -y python3-vtk7
