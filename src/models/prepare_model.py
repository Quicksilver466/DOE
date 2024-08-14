from src.models.modeling_phi3ex import Phi3exForCausalLM, transfer_phi3_weights, Phi3Config
from src.models.prepare_tokenizer import get_tokenizer
from transformers import AutoModelForCausalLM
from src.utils.utils import load_configs
import torch
import logging
import os

INFO_LOGGER = logging.getLogger("DOE-Info")
ERROR_LOGGER = logging.getLogger("DOE-Error")

CONFIGS = load_configs("model")

def get_phi3ex_model_for_inference(
        model_config,
        model_path: str=os.path.join(CONFIGS.get("untrained_model_save_base_path"), CONFIGS.get("untrained_model_save_name"))
):
    INFO_LOGGER.info("Loading Model")
    model = Phi3exForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)
    INFO_LOGGER.info("Model set")

    return model

def get_phi3ex_config(base_model_path="./tmp/models/Phi-3-mini-128k-instruct"):
    INFO_LOGGER.info("Setting up Phi-3 config")
    tokenizer = get_tokenizer()
    phi3_config = Phi3Config.from_pretrained(base_model_path)
    phi3_config.num_local_experts = CONFIGS.get("num_local_experts")
    phi3_config.threshold = CONFIGS.get("threshold")
    phi3_config.ar_loss_weight = CONFIGS.get("ar_loss_weight", 0.7)
    phi3_config.gating_loss_weight = CONFIGS.get("gating_loss_weight", 0.3)
    phi3_config.vocab_size = len(tokenizer)

    return phi3_config

def get_model_for_training(model_path="./tmp/models/Phi-3-mini-128k-instruct"):
    INFO_LOGGER.info("Setting up Phi-3ex Model")

    phi3_config = get_phi3ex_config(model_path)

    INFO_LOGGER.info("Loading original model")
    old_model = AutoModelForCausalLM.from_pretrained(model_path, config=phi3_config)
    tokenizer = get_tokenizer(model_path)

    INFO_LOGGER.info("Loading new model")
    new_model = Phi3exForCausalLM(phi3_config)

    INFO_LOGGER.info(f"Transferring weights from old model to new model with {phi3_config.num_local_experts} experts")
    new_model = transfer_phi3_weights(old_model, new_model, phi3_config.num_local_experts)

    INFO_LOGGER.info("Resetting embedding layer size and setting [CLS] token embedding to zero")
    new_model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        new_model.model.embed_tokens.weight[tokenizer.cls_token_id] = torch.zeros(phi3_config.hidden_size)

    INFO_LOGGER.info("Moving model to GPU")
    new_model = new_model.to(torch.device(0))

    INFO_LOGGER.info("Enabling gradient checkpointing")
    new_model.gradient_checkpointing_enable()

    INFO_LOGGER.info("Deleting old model")
    del old_model

    untrained_model_save_path = os.path.join(CONFIGS.get("untrained_model_save_base_path"), CONFIGS.get("untrained_model_save_name"))
    INFO_LOGGER.info(f"Saving untrained model at: {untrained_model_save_path}")
    new_model.save_pretrained(untrained_model_save_path)

    INFO_LOGGER.info("Phi-3ex Model set")

    return new_model