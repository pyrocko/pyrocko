FROM pyrocko-nest
COPY pyrocko.git /pyrocko.git
WORKDIR /src
RUN git clone /pyrocko.git pyrocko \
    && cd pyrocko && pip3 install --no-deps --break-system-packages .
