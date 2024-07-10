from datasets import load_dataset, load_from_disk
from datasets import disable_caching
from src.datasets.sft_datasets_transform import code_feedback_transform, maths_transform, medic_transform, mult_transform
from src.datasets.sft_datasets_transform import code_feedback_transform_phi3, maths_transform_phi3, medic_transform_phi3, mult_transform_phi3
from src.datasets.generic_datasets_transforms import chatml_transform, tokenize_transform
import os

disable_caching()

def exists_else_create(path):
    if os.path.exists(path):
        return
    
    os.mkdir(path)

def std_transformation():
    datasets = {
        "m-a-p/CodeFeedback-Filtered-Instruction": "/data/Datasets-LLMS/CodeFeedback-Filtered-Instruction",
        "TIGER-Lab/MathInstruct": "/data/Datasets-LLMS/MathInstruct",
        "lavita/ChatDoctor-HealthCareMagic-100k": "/data/Datasets-LLMS/ChatDoctor-HealthCareMagic-100k",
        "fnlp/moss-002-sft-data": "/data/Datasets-LLMS/moss-002-sft-data"
    }

    dataset = load_dataset(datasets["m-a-p/CodeFeedback-Filtered-Instruction"], split="train")
    result = dataset.map(code_feedback_transform, remove_columns=["query", "answer", "resource", "lang"])
    result.save_to_disk("/data/Datasets-LLMS/CodeFeedback-Filtered-Instruction-Transformed")

    dataset = load_dataset(datasets["TIGER-Lab/MathInstruct"], split="train")
    result = dataset.map(maths_transform, remove_columns=["source", "output", "instruction"])
    result.save_to_disk("/data/Datasets-LLMS/MathInstruct-Transformed")

    dataset = load_dataset(datasets["lavita/ChatDoctor-HealthCareMagic-100k"], split="train")
    result = dataset.map(medic_transform, remove_columns=["input", "output", "instruction"])
    result.save_to_disk("/data/Datasets-LLMS/ChatDoctor-HealthCareMagic-100k-Transformed")

    dataset = load_dataset(datasets["fnlp/moss-002-sft-data"], split="train")
    result = dataset.map(mult_transform, remove_columns=["num_turns", "plain_text", "prefix", "id"])
    result.save_to_disk("/data/Datasets-LLMS/moss-002-sft-data-Transformed")

def phi3_transformation():
    datasets = {
        "m-a-p/CodeFeedback-Filtered-Instruction": "/data/Datasets-LLMS/CodeFeedback-Filtered-Instruction",
        "TIGER-Lab/MathInstruct": "/data/Datasets-LLMS/MathInstruct",
        "lavita/ChatDoctor-HealthCareMagic-100k": "/data/Datasets-LLMS/ChatDoctor-HealthCareMagic-100k",
        "fnlp/moss-002-sft-data": "/data/Datasets-LLMS/moss-002-sft-data"
    }

    dataset = load_dataset(datasets["m-a-p/CodeFeedback-Filtered-Instruction"], split="train")
    result = dataset.map(code_feedback_transform_phi3, remove_columns=["query", "answer", "resource", "lang"])
    result.save_to_disk("/data/Datasets-LLMS/Transformed-Phi-3-Datasets/CodeFeedback-Filtered-Instruction-Transformed-phi3")

    dataset = load_dataset(datasets["TIGER-Lab/MathInstruct"], split="train")
    result = dataset.map(maths_transform_phi3, remove_columns=["source", "output", "instruction"])
    result.save_to_disk("/data/Datasets-LLMS/Transformed-Phi-3-Datasets/MathInstruct-Transformed-phi3")

    dataset = load_dataset(datasets["lavita/ChatDoctor-HealthCareMagic-100k"], split="train")
    result = dataset.map(medic_transform_phi3, remove_columns=["input", "output", "instruction"])
    result.save_to_disk("/data/Datasets-LLMS/Transformed-Phi-3-Datasets/ChatDoctor-HealthCareMagic-100k-Transformed-phi3")

    dataset = load_dataset(datasets["fnlp/moss-002-sft-data"], split="train")
    result = dataset.map(mult_transform_phi3, remove_columns=["num_turns", "plain_text", "prefix", "id"])
    result.save_to_disk("/data/Datasets-LLMS/Transformed-Phi-3-Datasets/moss-002-sft-data-Transformed-phi3")

def apply_generic_func(
        datasets_base_path="/data/Datasets-LLMS/Transformed-Phi-3-Datasets", 
        datasets_base_save_path="/data/Datasets-LLMS/Chatml-CLS-Phi-3-Datasets",
        func_to_map=chatml_transform
):
    exists_else_create(datasets_base_save_path)

    datasets = {
        "m-a-p/CodeFeedback-Filtered-Instruction": os.path.join(datasets_base_path, "CodeFeedback-Filtered-Instruction-Transformed-phi3"),
        "TIGER-Lab/MathInstruct": os.path.join(datasets_base_path, "MathInstruct-Transformed-phi3"),
        "lavita/ChatDoctor-HealthCareMagic-100k": os.path.join(datasets_base_path, "ChatDoctor-HealthCareMagic-100k-Transformed-phi3"),
        "fnlp/moss-002-sft-data": os.path.join(datasets_base_path, "moss-002-sft-data-Transformed-phi3")
    }

    for dataset_name, dataset_path in datasets.items():
        dataset = load_from_disk(dataset_path)
        results = dataset.map(func_to_map)
        results.save_to_disk(os.path.join(datasets_base_save_path, f"{dataset_name.split('/')[-1]}-Transformed-phi3"))