# Set the base image to Ubuntu
FROM ubuntu:20.04

# Set environment variables for package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install essential packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    gcc \
    clang \
    wget \
    git \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python package versions
RUN pip3 install --upgrade pip

# Install required Python packages
RUN pip3 install "torch>=1.7,<2" "click>=7.1,<8" "r2pipe>=1.5,<2"

RUN pip3 install tqdm
RUN pip3 install numpy

# Set the working directory
WORKDIR /root

COPY . .

# Set bash as the default command
CMD ["/bin/bash"]