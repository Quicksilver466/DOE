from src.utils.utils import load_configs
from src.models.prepare_tokenizer import get_tokenizer
import logging

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
        self.info_logger = logging.getLogger("DOE-Info")
        self.info_logger.setLevel(logging.INFO)

        self.error_logger = logging.getLogger("DOE-Error")
        self.error_logger.setLevel(logging.ERROR)

        self.info_logger.info("Setting Global Variables")

        self.dataset_configs = load_configs("dataset")
        self.model_configs = load_configs("model")
        self.trainer_configs = load_configs("trainer")
        self.lora_configs = load_configs("lora")
        self.sft_trainer_configs = load_configs("sft-trainer")
        self.tokenizer = get_tokenizer()

        self.info_logger.info("Global Variables Set")
        
    def get_gv(self):
        return {
            "INFO_LOGGER": self.info_logger,
            "ERROR_LOGGER": self.error_logger,
            "TOKENIZER": self.tokenizer,
            "DATASET_CONFIGS": self.dataset_configs,
            "TRAINER_CONFIGS": self.trainer_configs,
            "LORA_CONFIGS": self.lora_configs,
            "SFT-TRAINER-CONFIGS": self.sft_trainer_configs
        }