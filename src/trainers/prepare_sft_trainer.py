from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from src.utils.utils import load_configs
from datetime import datetime
from src.models.modeling_phi3ex import Phi3exForCausalLM
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging

INFO_LOGGER = logging.getLogger("DOE-Info")
ERROR_LOGGER = logging.getLogger("DOE-Error")

def get_sft_trainer(model: Phi3exForCausalLM, tokenizer: PreTrainedTokenizer, dataset: Dataset) -> SFTTrainer:
    INFO_LOGGER.info("Loading up configs for sft-trainer")
    sft_trainer_configs = load_configs("sft-trainer")
    
    INFO_LOGGER.info("Loading LORA configs")
    lora_config_dict = load_configs(config_type="lora")
    lora_config = LoraConfig(**lora_config_dict)

    INFO_LOGGER.info("Loading trainer arguments")
    training_arguments_dict = load_configs(config_type="trainer")
    training_arguments = TrainingArguments(**training_arguments_dict)
    training_arguments.run_name=f"DOE-phi3ex-sft-LORA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"

    INFO_LOGGER.info("Instantiating SFT-Trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field=sft_trainer_configs.get("dataset_text_field", "text"),
        peft_config=lora_config,
        packing=sft_trainer_configs.get("packing", False),
        max_seq_length=sft_trainer_configs.get("max_seq_length", 720)
    )

    INFO_LOGGER.info("SFT-Trainer set")

    return trainer