ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.11-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/nnunet_pyt
WORKDIR /workspace/nnunet_pyt

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt
