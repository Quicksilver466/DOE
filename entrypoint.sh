#! /bin/bash

set -e

huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

datasets_base_path="./tmp/datasets"
models_base_path="./tmp/models"

dataset_name="Quicksilver1/DOE-Merged-Tokenized-v1"
model_name="microsoft/Phi-3-mini-128k-instruct"

dataset_filename="${dataset_name##*/}"
model_filename="${model_name##*/}"

dataset_complete_path="${datasets_base_path}/${dataset_filename}"
model_complete_path="${models_base_path}/${model_filename}"

mkdir -p "$dataset_complete_path"
mkdir -p "$model_complete_path"

huggingface-cli download $dataset_name --repo-type dataset --local-dir $dataset_complete_path
huggingface-cli download $model_name --local-dir $model_complete_path

python trainer.py