docker build\
    -t sentiment-analyzer-hf:${2}-${3}\
    --build-arg HF_ID=$1\
    --build-arg MODEL_NAME=$2\
    --build-arg MODEL_VERSION=$3\