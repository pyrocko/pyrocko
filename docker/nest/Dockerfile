ARG base_image

FROM $base_image
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get upgrade -y

# env requirements
RUN apt-get install -y python3-pip twine
RUN apt-get install -y wget

# linter requirements
RUN apt-get install -y flake8

# build requirements
RUN apt-get install -y make git
RUN apt-get install -y python3-dev python3-setuptools python3-numpy-dev

RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# testing requirements
RUN apt-get install -y xvfb libgles2-mesa
RUN apt-get install -y python3-coverage python3-pytest

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

# optional runtime requirements
RUN apt-get install -y \
        python3-jinja2 python3-pybtex


# optional runtime requirements obspy, kite
RUN apt-get install -y \
        python3-geojson python3-pyqtgraph \
    && pip3 install --no-deps \
        utm git+https://github.com/pyrocko/kite

ARG base_image
RUN [ "$base_image" != "debian:10" ] \
    && [ "$base_image" != "ubuntu:20.04" ] \
    && ( apt-get install -y \
        python3-sqlalchemy python3-lxml python3-greenlet \
    && pip3 install --no-deps obspy==1.3 || exit 1 ) || :  # 2023-01-16: 1.4 has too high numpy requirement for debian:stabl

# additional runtime requirements for gmt
RUN apt-get install -y \
        gmt gmt-gshhg poppler-utils imagemagick

# additional runtime requirements for fomosto backends
RUN apt-get install -y autoconf gfortran
WORKDIR /src
RUN git clone https://git.pyrocko.org/pyrocko/fomosto-qseis.git \
    && cd fomosto-qseis && autoreconf -i && ./configure && make && make install
WORKDIR /src
RUN git clone https://git.pyrocko.org/pyrocko/fomosto-psgrn-pscmp.git \
    && cd fomosto-psgrn-pscmp && autoreconf -i && ./configure && make && make install
WORKDIR /src
RUN git clone https://git.pyrocko.org/pyrocko/fomosto-qseis2d.git \
    && cd fomosto-qseis2d && autoreconf -i && ./configure && make && make install
WORKDIR /src
RUN git clone https://git.pyrocko.org/pyrocko/fomosto-qssp.git \
    && cd fomosto-qssp && autoreconf -i && ./configure && make && make install
WORKDIR /src
RUN git clone https://git.pyrocko.org/pyrocko/fomosto-qssp2017.git \
    && cd fomosto-qssp2017 && autoreconf -i && ./configure && make && make install

