FROM pyrocko-nest

WORKDIR /src
RUN git clone https://git.pyrocko.org/pyrocko/pyrocko.git \
    && cd pyrocko && python2 setup.py install && python3 setup.py install
