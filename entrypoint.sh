#! /bin/bash

set -e

curl -o gitlfs.tar.gz -L https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar -xvzf gitlfs.tar.gz
git lfs install

cd ./code

makedir -p ./tmp/datasets
cd ./tmp/datasets
git clone https://huggingface.co/datasets/Quicksilver1/DOE-Merged-Tokenized-v1
cd ..
cd ..

makedir -p ./tmp/models
cd ./tmp/models
git clone https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
cd ..
cd ..

echo success