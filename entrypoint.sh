#! /bin/bash

set -e

huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

mkdir -p ./tmp/datasets/DOE-Merged-Tokenized-v1
mkdir -p ./tmp/models/Phi-3-mini-128k-instruct

huggingface-cli download Quicksilver1/DOE-Merged-Tokenized-v1 --repo-type dataset --local-dir /code/tmp/datasets/DOE-Merged-Tokenized-v1
huggingface-cli download microsoft/Phi-3-mini-128k-instruct --local-dir /code/tmp/models/Phi-3-mini-128k-instruct

echo success