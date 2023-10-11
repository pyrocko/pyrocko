FROM pyrocko-nest-debian-12
ENV DEBIAN_FRONTEND=noninteractive

# docs requirements
RUN apt-get install -y \
    texlive-fonts-recommended texlive-latex-extra \
    texlive-latex-recommended python3-sphinx \
    && pip3 install sphinx-qt-documentation \
    git+https://git.pyrocko.org/pyrocko/sphinx-sleekcat-theme.git
