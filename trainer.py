from src.utils.glob_vars import GlobalVars
import mlflow
import os
from peft import AutoPeftModelForCausalLM
from datetime import datetime
from src.utils.utils import create_dir_if_not_exists
import gc
import torch

GV = GlobalVars()

def train(model_save_path="./tmp/models/DOE-SFT"):
    model_save_base_path = "/".join(model_save_path.split("/")[:-1])
    model_save_name = model_save_path.split("/")[-1]
    create_dir_if_not_exists(model_save_base_path)

    mlflow.set_experiment(experiment_id=os.environ.get("MLFLOW_EXPERIMENT_ID"))

    GV.get_gv().get("INFO_LOGGER").info("Starting the Training")

    GV.get_gv().get("SFT_TRAINER").train()
    
    GV.get_gv().get("INFO_LOGGER").info("Training Finished")

    GV.get_gv().get("INFO_LOGGER").info("Logging final parameters and model to MLFlow")
    last_run_id = mlflow.last_active_run().info.run_id
    with mlflow.start_run(run_id=last_run_id):
        mlflow.log_params(GV.get_gv().get("LORA_CONFIGS"))
        mlflow.transformers.log_model(
            {
                "model": GV.get_gv().get("SFT_TRAINER").model, 
                "tokenizer": GV.get_gv().get("TOKENIZER")
            }, 
            artifact_path="model", 
            task="text-generation"
        )

    GV.get_gv().get("INFO_LOGGER").info("Saving unmerged model through trainer")
    GV.get_gv().get("SFT_TRAINER").save_model(model_save_path)
    
    del GV.model
    del GV.sft_trainer
    gc.collect()
    torch.cuda.empty_cache()

    GV.get_gv().get("INFO_LOGGER").info("Loading Unmerged model for merger")
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_save_path,
        adapter_name="sft",
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    GV.get_gv().get("INFO_LOGGER").info("Saving Merged Model")
    merged_model_save_path = os.path.join(model_save_base_path, f"Merged-{model_save_name}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        merged_model_save_path,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    GV.get_gv().get("INFO_LOGGER").info("Training Pipeline Done")

if __name__ == "__main__":
    train(model_save_path=f"./tmp/models/DOE-SFT-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}")