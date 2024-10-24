from src.utils.glob_vars import GlobalVars
from datasets import Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

GV = GlobalVars()

def chatml_transform(example, add_cls_token=True, **kwargs):
    messages = example["text"]
    chatml_messages = GV.get_gv().get("TOKENIZER").apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    chatml_messages.append(GV.get_gv().get("TOKENIZER").eos_token_id)
    chatml_messages = [GV.get_gv().get("TOKENIZER").cls_token_id] + chatml_messages if add_cls_token else chatml_messages

    return {"text": GV.get_gv().get("TOKENIZER").decode(chatml_messages)}

def tokenize_transform(example, expert_indices, add_cls_token=False) -> dict:
    output = GV.get_gv().get("TOKENIZER")(
        example["text"],
        add_special_tokens=True,
        truncation=GV.get_gv().get("DATASET_CONFIGS").get("truncation"),
        padding=GV.get_gv().get("DATASET_CONFIGS").get("padding"),
        max_length=GV.get_gv().get("DATASET_CONFIGS").get("max_length"),
        return_overflowing_tokens=False,
        return_length=False
    )

    if add_cls_token:
        output["input_ids"].append(GV.get_gv().get("TOKENIZER").cls_token_id)
        output["attention_mask"].insert(0, 1)

    return {
        "input_ids": output["input_ids"],
        "attention_mask": output["attention_mask"],
        "expert_indices": expert_indices
    }

def concat_shuffle_datasets(datasets: list[Dataset]):
    merged_dataset: Dataset = concatenate_datasets(datasets)
    shuffled_dataset = merged_dataset.shuffle()
    shuffled_dataset.flatten_indices()

    return shuffled_dataset

def flatten_expert_indices(example):
    example["expert_indices"] = example["expert_indices"][0]
    return example

def custom_tokenize_transform(example, expert_indices, tokenizer: PreTrainedTokenizer, add_special_tokens: bool, add_cls_token=False) -> dict:
    output = tokenizer(
        example["text"],
        add_special_tokens=add_special_tokens,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_overflowing_tokens=False,
        return_length=False
    )

    if add_cls_token:
        output["input_ids"].insert(0, tokenizer.cls_token_id)
        output["attention_mask"].insert(0, 1)

    return {
        "input_ids": output["input_ids"],
        "attention_mask": output["attention_mask"],
        "expert_indices": expert_indices
    }

def custom_chatml_transform(example, tokenizer: PreTrainedTokenizer, add_cls_token=True, **kwargs) -> dict:
    messages = []

    for conversation in example.get("conversations"):
        message = {}
        if(conversation.get("from")=="human"):
            role = "user"
        elif(conversation.get("from")=="gpt"):
            role = "assistant"
        else:
            role = "system"

        message["role"] = role
        message["content"] = conversation.get("value")
        #message["weight"] = conversation.get("weight")

        messages.append(message)


    chatml_messages = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    chatml_messages.append(tokenizer.eos_token_id)
    chatml_messages = [tokenizer.cls_token_id] + chatml_messages if add_cls_token else chatml_messages

    return {"text": tokenizer.decode(chatml_messages)}