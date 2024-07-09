from transformers import AutoTokenizer
from src.utils.glob_vars import GlobalVars

GV = GlobalVars()

def get_tokenizer(tokenizer_path="/data/LLM-weights/Phi-3-mini-128k-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=True)
    return tokenizer

TOKENIZER = get_tokenizer()

def chatml_transform(example):
    messages = example["text"]
    chatml_messages = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    chatml_messages.append(TOKENIZER.eos_token_id)

    return {"text": TOKENIZER.decode(chatml_messages)}

def add_CLS_token(example):
    pass

def tokenzie_transform(example):
    output = TOKENIZER(
        example["text"],
        add_special_tokens=True,
        truncation=GV.get_gv().get("DATASET_CONFIGS").get("truncation"),
        padding=GV.get_gv().get("DATASET_CONFIGS").get("padding"),
        max_length=GV.get_gv().get("DATASET_CONFIGS").get("max_length"),
        return_overflowing_tokens=False,
        return_length=False
    )

    return {
        "input_ids": output["input_ids"],
        "attention_mask": output["attention_mask"]
    }