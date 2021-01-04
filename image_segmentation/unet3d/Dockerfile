#ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

#RUN apt-get update && \
#    apt-get upgrade -y && \
#    apt-get install -y git

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly nvidia-dali-nightly-cuda110