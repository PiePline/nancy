FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04

ARG APP_DIR=/app

WORKDIR $APP_DIR
ADD . $APP_DIR

RUN apt-get update && apt-get -y install python3.8 python3-pip && pip3 install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt && python3 -m pip install dvc

RUN dvc pull && dvc repro

RUN rm -rf artifscts/dataset artifacts/indices