from datasets import load_from_disk

def get_dataset(dataset_path="./tmp/datasets/Merged-Shuffled-phi3-tokenized"):
    return load_from_disk(dataset_path=dataset_path)