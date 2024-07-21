from datasets import load_from_disk

def get_dataset(dataset_path="./tmp/datasets/DOE-Merged-Tokenized-v1"):
    return load_from_disk(dataset_path=dataset_path)