FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Update packages and install dependencies
RUN apt-get update && apt-get install -y \
    wget unzip build-essential cmake git g++ make \
    python3 python3-pip zlib1g-dev libjpeg-dev libpng-dev \
    libgtk2.0-dev pkg-config

# Download and install libtorch
RUN wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip -d /opt && \
   rm -rf libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip

# Set env variables to use libtorch
ENV Torch_DIR=/opt/libtorch/share/cmake/Torch

# Install torchvision from source
RUN git clone --branch release/0.14 https://github.com/pytorch/vision.git && \
    cd vision && mkdir build && cd build && cmake .. \
    	-DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=on \
    && make && make install DESTDIR=/opt/torchvision && \
    cd ../.. && rm -rf vision

# Set env variables to use torchvision
ENV TorchVision_DIR=/opt/torchvision/usr/local/share/cmake/TorchVision

# Install packages for processing video
RUN apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev

# Install OpenCV from source
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    unzip opencv.zip && rm opencv.zip && cd opencv-4.x && \
    mkdir build && cd build && cmake .. && make -j4 && \
    make install DESTDIR=/opt/opencv && cd ../.. && rm -rf opencv-4.x

# Set env variables to use OpenCV
ENV OpenCV_DIR=/opt/opencv/usr/local/lib/cmake/opencv4
ENV LD_LIBRARY_PATH=/opt/opencv/usr/local/lib:$LD_LIBRARY_PATH

# Copy source code to container
WORKDIR /container-ocr
COPY . /container-ocr

