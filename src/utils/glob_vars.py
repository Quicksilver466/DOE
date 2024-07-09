import os
from src.utils.utils import load_configs

def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@singleton
class GlobalVars:
    def __init__(self) -> None:
        self.configs_base_path = "./src/configs"
        self.set_gv()

    def set_gv(self):
        self.dataset_configs = load_configs(os.path.join(self.configs_base_path, "dataset-configs.json"))
        
    def get_gv(self):
        return {
            "DATASET_CONFIGS": self.dataset_configs
        }