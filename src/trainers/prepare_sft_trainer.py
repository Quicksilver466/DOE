from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from src.utils.utils import load_configs
from datetime import datetime
from src.models.modeling_phi3ex import Phi3exForCausalLM
from datasets import Dataset
from transformers import PreTrainedTokenizer

def get_sft_trainer(model: Phi3exForCausalLM, tokenizer: PreTrainedTokenizer, dataset: Dataset) -> SFTTrainer:
    sft_trainer_configs = load_configs("sft-trainer")
    
    lora_config_dict = load_configs(config_type="lora")
    lora_config = LoraConfig(**lora_config_dict)

    training_arguments_dict = load_configs(config_type="trainer")
    training_arguments = TrainingArguments(**training_arguments_dict)
    training_arguments.run_name=f"DOE-phi3ex-sft-LORA-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}"

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

    return trainer