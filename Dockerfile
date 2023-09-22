FROM ubuntu:latest

WORKDIR /app

# install wget
# @TODO: split
RUN apt-get update && apt-get -y install wget

# install CUDA toolkit 11.8
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
# @TODO:REMOVE
RUN apt-get update

# set noninteractive frontend, otherwise it asks for "country of origin for the keyboard"
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda

# install pip
RUN apt-get -y install python3-pip

# install torch with CUDA 11.8 support
RUN pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# copy and install requirements
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# @TODO: move to top
# @ TODO: is it needed?
# install NVIDIA dkms driver (above CUDA toolkit installation already installs nvidia-driver-535)
RUN apt-get install nvidia-dkms-535

CMD ["nvidia-smi"]
#CMD ["dkms", "status"]

COPY ChessAlphaZero ChessAlphaZero

# install Python 3.10
RUN apt-get -y install python3.10

# -u flag for unbuffered output, otherwise python output gets buffered
# inside docker container and only gets flushed when the script exits
#CMD ["python3.10", "-u", "ChessAlphaZero/AlphaZeroDP.py"]