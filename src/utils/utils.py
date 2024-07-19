import json
from typing import Literal
import os

CONFIGS_BASE_PATH = "./src/configs"
CONFIGS_FILENAMES = {
    "dataset": "dataset-configs.json",
    "model": "model-configs.json",
    "trainer": "trainer-configs.json",
    "lora": "lora-configs.json",
    "sft-trainer": "sft-trainer-configs.json"
}

def load_configs(config_type: Literal["dataset", "model", "trainer", "lora", "sft-trainer"]) -> dict:
    config_path = os.path.join(CONFIGS_BASE_PATH, CONFIGS_FILENAMES.get(config_type))
    dataset_configs = json.load(open(config_path, "r"))
    return dataset_configs

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set): return list(obj)
        return json.JSONEncoder.default(self, obj)
    
def save_config(config: dict, config_type: Literal["dataset", "model", "trainer", "lora"], use_set_encoder=False) -> bool:
    """Save any config to json pre-configured files

    Args:
        config (dict): The config to save as a Dictionary
        config_type (Literal[dataset, model, trainer, lora]): Type of config to save
        use_set_encoder (bool, optional): To set to true incase the config consists of any sets. Parameter included for saving lora config which can have sets
        Defaults to False.

    Returns:
        bool: Returns whether saving operation was successful or not. Incase it isn't successful, logger will log appropriate exception
    """
    try:
        config_path = os.path.join(CONFIGS_BASE_PATH, CONFIGS_FILENAMES.get(config_type))
        cls = SetEncoder if use_set_encoder else None
        json.dump(config, open(config_path, "w"), cls=cls)
        return True
    except BaseException as e:
        print(e)
        return False