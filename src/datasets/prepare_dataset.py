from datasets import load_from_disk
import logging

INFO_LOGGER = logging.getLogger("DOE-Info")
ERROR_LOGGER = logging.getLogger("DOE-Error")

def get_dataset(dataset_path="./tmp/datasets/DOE-Merged-Tokenized-v1"):
    try:
        INFO_LOGGER.info("Loading Dataset")
        dataset = load_from_disk(dataset_path=dataset_path)
        INFO_LOGGER.info("Dataset Loaded")
        return dataset
    
    except BaseException as e:
        ERROR_LOGGER.exception(f"Couldn't load dataset because of the following error: {e}")