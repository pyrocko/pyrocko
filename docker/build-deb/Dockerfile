ARG base_image

FROM $base_image
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install -y debhelper-compat dh-python python3-all python3-dev python3-numpy git python3-setuptools debmake
