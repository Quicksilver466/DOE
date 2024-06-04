from datasets import load_dataset
from datasets import disable_caching
from src.datasets.datasets_transform import code_feedback_transform, maths_transform, medic_transform, mult_transform
from src.datasets.datasets_transform import code_feedback_transform_phi3, maths_transform_phi3, medic_transform_phi3, mult_transform_phi3

disable_caching()

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

    #dataset = load_dataset(datasets["TIGER-Lab/MathInstruct"], split="train")
    #result = dataset.map(maths_transform, remove_columns=["source", "output", "instruction"])
    #result.save_to_disk("/data/Datasets-LLMS/MathInstruct-Transformed")
#
    #dataset = load_dataset(datasets["lavita/ChatDoctor-HealthCareMagic-100k"], split="train")
    #result = dataset.map(medic_transform, remove_columns=["input", "output", "instruction"])
    #result.save_to_disk("/data/Datasets-LLMS/ChatDoctor-HealthCareMagic-100k-Transformed")
#
    #dataset = load_dataset(datasets["fnlp/moss-002-sft-data"], split="train")
    #result = dataset.map(mult_transform, remove_columns=["num_turns", "plain_text", "prefix", "id"])
    #result.save_to_disk("/data/Datasets-LLMS/moss-002-sft-data-Transformed")

def phi3_transformation():
    datasets = {
        "m-a-p/CodeFeedback-Filtered-Instruction": "/data/Datasets-LLMS/CodeFeedback-Filtered-Instruction",
        "TIGER-Lab/MathInstruct": "/data/Datasets-LLMS/MathInstruct",
        "lavita/ChatDoctor-HealthCareMagic-100k": "/data/Datasets-LLMS/ChatDoctor-HealthCareMagic-100k",
        "fnlp/moss-002-sft-data": "/data/Datasets-LLMS/moss-002-sft-data"
    }

    dataset = load_dataset(datasets["m-a-p/CodeFeedback-Filtered-Instruction"], split="train")
    result = dataset.map(code_feedback_transform_phi3, remove_columns=["query", "answer", "resource", "lang"])
    result.save_to_disk("/data/Datasets-LLMS/CodeFeedback-Filtered-Instruction-Transformed-phi3")

    #dataset = load_dataset(datasets["TIGER-Lab/MathInstruct"], split="train")
    #result = dataset.map(maths_transform_phi3, remove_columns=["source", "output", "instruction"])
    #result.save_to_disk("/data/Datasets-LLMS/MathInstruct-Transformed-phi3")
#
    #dataset = load_dataset(datasets["lavita/ChatDoctor-HealthCareMagic-100k"], split="train")
    #result = dataset.map(medic_transform_phi3, remove_columns=["input", "output", "instruction"])
    #result.save_to_disk("/data/Datasets-LLMS/ChatDoctor-HealthCareMagic-100k-Transformed-phi3")
#
    #dataset = load_dataset(datasets["fnlp/moss-002-sft-data"], split="train")
    #result = dataset.map(mult_transform_phi3, remove_columns=["num_turns", "plain_text", "prefix", "id"])
    #result.save_to_disk("/data/Datasets-LLMS/moss-002-sft-data-Transformed-phi3")

std_transformation()
phi3_transformation()