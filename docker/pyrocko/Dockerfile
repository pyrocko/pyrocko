FROM pyrocko-nest

WORKDIR /src
RUN git clone https://git.pyrocko.org/pyrocko/pyrocko.git \
    && cd pyrocko && python3 install.py user --yes
