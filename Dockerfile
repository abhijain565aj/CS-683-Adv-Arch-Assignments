FROM --platform=linux/amd64 ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    gcc-7 \
    g++-7 \
    make \
    git \
    && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 70 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-7 70

ENV CC=g++-7
ENV CXX=g++-7

WORKDIR /workspace

CMD ["/bin/bash"]

# docker build --platform linux/amd64 -t cs683-x86 .