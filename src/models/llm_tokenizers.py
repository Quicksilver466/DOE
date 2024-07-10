from transformers import AutoTokenizer

class SpecialTokens:
    cls_token = "[CLS]"
    pad_token = "<|pad|>"

def get_tokenizer(tokenizer_path="/data/LLM-weights/Phi-3-mini-128k-instruct"):
    chat_format_tokens = SpecialTokens
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_eos_token=True)

    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                chat_format_tokens.cls_token,
                chat_format_tokens.pad_token
            ]
        }
    )
    tokenizer.pad_token = chat_format_tokens.pad_token
    tokenizer.cls_token = chat_format_tokens.cls_token

    return tokenizer