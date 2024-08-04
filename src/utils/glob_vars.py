from src.utils.utils import load_configs
from src.models.prepare_tokenizer import get_tokenizer
from src.models.prepare_model import get_model_for_training
from src.datasets.prepare_dataset import get_dataset
from src.trainers.prepare_sft_trainer import get_sft_trainer
from src.utils.hooks import HookRegistry, logitloss_hook
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
        logging.basicConfig()

        self.info_logger = logging.getLogger("DOE-Info")
        self.info_logger.setLevel(logging.INFO)

        self.error_logger = logging.getLogger("DOE-Error")
        self.error_logger.setLevel(logging.ERROR)

        self.info_logger.info("Setting Global Variables")

        self.info_logger.info("Loading all the Configs")
        self.dataset_configs = load_configs("dataset")
        self.model_configs = load_configs("model")
        self.trainer_configs = load_configs("trainer")
        self.lora_configs = load_configs("lora")
        self.sft_trainer_configs = load_configs("sft-trainer")
        self.info_logger.info("All configs loaded")
        self.dataset = get_dataset()
        self.tokenizer = get_tokenizer()
        self.model = get_model_for_training()
        #self.hook_registry = HookRegistry()
        #self.hook_registry.add_hook(self.model.gating_loss_fct, logitloss_hook)
        #self.hook_registry.register_hooks()
        self.sft_trainer = get_sft_trainer(self.model, self.tokenizer, self.dataset)

        self.info_logger.info("Global Variables Set")
        
    def get_gv(self):
        return {
            "INFO_LOGGER": self.info_logger,
            "ERROR_LOGGER": self.error_logger,
            "TOKENIZER": self.tokenizer,
            "DATASET_CONFIGS": self.dataset_configs,
            "TRAINER_CONFIGS": self.trainer_configs,
            "LORA_CONFIGS": self.lora_configs,
            "SFT_TRAINER_CONFIGS": self.sft_trainer_configs,
            "DATASET": self.dataset,
            "MODEL": self.model,
            "SFT_TRAINER": self.sft_trainer
        }