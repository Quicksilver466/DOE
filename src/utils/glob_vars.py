from src.utils.utils import load_configs
from src.models.prepare_tokenizer import get_tokenizer

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
        self.set_gv()

    def set_gv(self):
        self.dataset_configs = load_configs("dataset")
        self.model_configs = load_configs("model")
        self.trainer_configs = load_configs("trainer")
        self.lora_configs = load_configs("lora")
        self.tokenizer = get_tokenizer()
        
    def get_gv(self):
        return {
            "TOKENIZER": self.tokenizer,
            "DATASET_CONFIGS": self.dataset_configs
        }