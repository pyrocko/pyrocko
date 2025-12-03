ARG base_image

FROM $base_image
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y debhelper-compat dh-python debmake python3-all python3-dev python3-numpy python3-setuptools pybuild-plugin-pyproject python3-venv python3-pip git
