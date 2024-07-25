from transformers import AutoTokenizer
import logging

INFO_LOGGER = logging.getLogger("DOE-Info")
ERROR_LOGGER = logging.getLogger("DOE-Error")

class SpecialTokens:
    cls_token = "[CLS]"
    pad_token = "<|pad|>"

def get_tokenizer(tokenizer_path="./tmp/models/Phi-3-mini-128k-instruct"):
    try:
        INFO_LOGGER.info("Setting up Tokenizer")
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
    
        INFO_LOGGER.info("Tokenizer is set")

        return tokenizer
    
    except BaseException as e:
        ERROR_LOGGER.exception(f"Couldn't setup Tokenizer because of the following exception: {e}")