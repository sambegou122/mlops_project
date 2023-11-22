#!/bin/bash

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --model_version) model_version="$2"; shift ;;
        --status) status="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if all arguments are provided
if [ -z "$model_name" ] || [ -z "$model_version" ] || [ -z "$status" ]; then
    echo "Usage: ./promote.sh --model_name <model_name> --model_version <model_version> --status <status>"
    exit 1
fi

# Promote the model
mlflow models transition-to-stage -n "$model_name" -v "$model_version" -s "$status"