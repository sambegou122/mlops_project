#!/bin/sh
docker build\
    -t sentiment-analyzer:${2}-${3}\
    --build-arg MLFLOW_SERVER_URI=$1\
    --build-arg MODEL_NAME=$2\
    --build-arg MODEL_VERSION=$3\
    .
