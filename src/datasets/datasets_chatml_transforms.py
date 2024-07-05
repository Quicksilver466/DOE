from transformers import AutoTokenizer

def get_tokenizer(tokenizer_path="/data/LLM-weights/Phi-3-mini-128k-instruct"):
    return AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=True)

TOKENIZER = get_tokenizer()

def chatml_transform(example):
    messages = example["text"]
    chatml_messages = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    chatml_messages.append(TOKENIZER.eos_token_id)

    return {"text": TOKENIZER.decode(chatml_messages)}