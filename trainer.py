from src.utils.glob_vars import GlobalVars
import mlflow
import os
from peft import PeftModel
from src.models.modeling_phi3ex import Phi3exForCausalLM
from datetime import datetime
from src.utils.utils import create_dir_if_not_exists, setup_mlflow, remove_dir
import gc
import torch
from huggingface_hub import HfApi

setup_mlflow()
GV = GlobalVars()
API = HfApi()

def train(model_save_path="./tmp/models/DOE-SFT"):
    model_save_base_path = "/".join(model_save_path.split("/")[:-1])
    model_save_name = model_save_path.split("/")[-1]
    create_dir_if_not_exists(model_save_base_path)

    GV.get_gv().get("INFO_LOGGER").info("Starting the Training")

    GV.get_gv().get("SFT_TRAINER").train()
    
    GV.get_gv().get("INFO_LOGGER").info("Training Finished")

    GV.get_gv().get("INFO_LOGGER").info("Saving unmerged model through trainer")
    GV.get_gv().get("SFT_TRAINER").save_model(model_save_path)
    
    del GV.model
    del GV.sft_trainer
    GV.model = None
    GV.sft_trainer = None
    gc.collect()
    torch.cuda.empty_cache()

    GV.get_gv().get("INFO_LOGGER").info("Loading Unmerged model for merger")
    base_model_path = os.path.join(
        GV.get_gv().get("MODEL_CONFIGS").get("untrained_model_save_base_path"),
        GV.get_gv().get("MODEL_CONFIGS").get("untrained_model_save_name")
    )
    GV.get_gv().get("INFO_LOGGER").info(f"Loading Base Model from path: {base_model_path}")
    base_model = Phi3exForCausalLM.from_pretrained(pretrained_model_name_or_path=base_model_path)
    peft_model = PeftModel.from_pretrained(
        base_model,
        model_save_path,
        adapter_name="sft",
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    merged_model_save_path = os.path.join(model_save_base_path, f"Merged-{model_save_name}")
    GV.get_gv().get("INFO_LOGGER").info(f"Saving Merged Model at path: {merged_model_save_path}")
    merged_model = peft_model.merge_and_unload()
    remove_dir(base_model_path)
    merged_model.save_pretrained(
        merged_model_save_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    try:
        API.create_repo(
            "Quicksilver1/DOE-Model-test",
            repo_type="model",
        )
    except:
        pass

    API.upload_folder(
        repo_id="Quicksilver1/DOE-Model-test",
        folder_path=merged_model_save_path,
        repo_type="model"
    )

    try:
        GV.get_gv().get("INFO_LOGGER").info("Logging final parameters and model to MLFlow")
        last_run_id = mlflow.last_active_run().info.run_id
        with mlflow.start_run(run_id=last_run_id):
            mlflow.log_params(GV.get_gv().get("LORA_CONFIGS"))
            mlflow.transformers.log_model(
                {
                    "model": merged_model, 
                    "tokenizer": GV.get_gv().get("TOKENIZER")
                }, 
                artifact_path="model", 
                task="text-generation",
                save_pretrained=True
            )
    except BaseException as e:
        GV.get_gv().get("ERROR_LOGGER").exception(f"Couldn't log model to MLFlow because of following error: \n{e}")

    GV.get_gv().get("INFO_LOGGER").info("Training Pipeline Done")

if __name__ == "__main__":
    train(model_save_path=f"/workspace/models/DOE-SFT-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}")