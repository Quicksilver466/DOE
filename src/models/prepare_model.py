from src.models.modeling_phi3ex import Phi3exForCausalLM, transfer_phi3_weights, Phi3Config
from src.models.prepare_tokenizer import get_tokenizer
from transformers import AutoModelForCausalLM
from src.utils.utils import load_configs
import torch

CONFIGS = load_configs("model")

def get_model_for_inference(model_path="/data/LLM-weights/Phi-3-mini-128k-instruct"):
    pass

def get_model_for_training(model_path="/data/LLM-weights/Phi-3-mini-128k-instruct"):
    old_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = get_tokenizer(model_path)

    phi3_config = Phi3Config.from_pretrained(model_path)
    phi3_config.num_local_experts = CONFIGS.get("num_local_experts")
    phi3_config.threshold = CONFIGS.get("threshold")

    new_model = Phi3exForCausalLM(phi3_config)

    new_model = transfer_phi3_weights(old_model, new_model, phi3_config.num_local_experts)

    with torch.no_grad():
        new_model.model.embed_tokens.weight[tokenizer.cls_token_id] = torch.zeros(phi3_config.hidden_size)

    del old_model
    return new_model