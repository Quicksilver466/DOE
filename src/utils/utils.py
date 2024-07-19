import json
from typing import Literal
import os

CONFIGS_BASE_PATH = "./src/configs"
CONFIGS_FILENAMES = {
    "dataset": "dataset-configs.json",
    "model": "model-configs.json",
    "trainer": "trainer-configs.json",
    "lora": "lora-configs.json"
}

def load_configs(config_type: Literal["dataset", "model", "trainer", "lora"]) -> dict:
    config_path = os.path.join(CONFIGS_BASE_PATH, CONFIGS_FILENAMES.get(config_type))
    dataset_configs = json.load(open(config_path, "r"))
    return dataset_configs