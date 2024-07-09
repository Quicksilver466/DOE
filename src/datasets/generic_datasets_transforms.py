from src.utils.glob_vars import GlobalVars

GV = GlobalVars()

def chatml_transform(example):
    messages = example["text"]
    chatml_messages = GV.get_gv().get("TOKENIZER").apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    chatml_messages.append(GV.get_gv().get("TOKENIZER").eos_token_id)

    return {"text": GV.get_gv().get("TOKENIZER").decode(chatml_messages)}

def tokenize_transform(example):
    output = GV.get_gv().get("TOKENIZER")(
        example["text"],
        add_special_tokens=False,
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