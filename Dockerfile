FROM nvcr.io/nvidia/l4t-cuda:12.2.12-runtime

# Install nvidia-l4t-core
RUN \
    echo "deb https://repo.download.nvidia.com/jetson/common r36.3 main" >> /etc/apt/sources.list && \
    echo "deb https://repo.download.nvidia.com/jetson/t234 r36.3 main" >> /etc/apt/sources.list && \
    apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
    mkdir -p /opt/nvidia/l4t-packages/ && \
    touch /opt/nvidia/l4t-packages/.nv-l4t-disable-boot-fw-update-in-preinstall

RUN apt-get update \
    && echo "Y" | apt-get install -y --no-install-recommends nvidia-l4t-core

ENV UDEV=1

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb \
  && dpkg -i cuda-keyring_1.1-1_all.deb \
  && apt-get update \
  && apt-get -y install cuda-toolkit-12-5

# Install necessary dependencies including gcc
RUN apt-get update \
    && apt-get install -y wget build-essential git cmake pkg-config curl vim python3 python3-pip docker-compose ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install GCC 12 and G++ 12
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-12 g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

##### Install necessary packages
COPY ./requirements.txt /
RUN pip3 install -r requirements.txt && rm -rf requirements.txt

# Install Google Test
RUN git clone https://github.com/google/googletest.git \
  && cd googletest \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j12 \
  && make -j12 install \
  && cd ../.. \
  && rm -rf googletest


