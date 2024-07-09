import json

def load_configs(config_path: str) -> dict:
    dataset_configs = json.load(open(config_path, "r"))
    return dataset_configs