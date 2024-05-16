from datasets import load_dataset
from datasets import disable_caching
from src.datasets.datasets_transform import code_feedback_transform, maths_transform, medic_transform, mult_transform

disable_caching()

DATASETS = {
    "m-a-p/CodeFeedback-Filtered-Instruction": "/usr/Datasets-LLMS/CodeFeedback-Filtered-Instruction",
    "TIGER-Lab/MathInstruct": "/usr/Datasets-LLMS/MathInstruct",
    "lavita/ChatDoctor-HealthCareMagic-100k": "/usr/Datasets-LLMS/ChatDoctor-HealthCareMagic-100k",
    "fnlp/moss-002-sft-data": "/usr/Datasets-LLMS/moss-002-sft-data"
}

dataset = load_dataset(DATASETS["m-a-p/CodeFeedback-Filtered-Instruction"], split="train")
result = dataset.map(code_feedback_transform, remove_columns=["query", "answer", "resource", "lang"])
result.save_to_disk("/usr/Datasets-LLMS/CodeFeedback-Filtered-Instruction-Transformed")

dataset = load_dataset(DATASETS["TIGER-Lab/MathInstruct"], split="train")
result = dataset.map(maths_transform, remove_columns=["source", "output", "instruction"])
result.save_to_disk("/usr/Datasets-LLMS/CodeFeedback-Filtered-Instruction-Transformed")

dataset = load_dataset(DATASETS["lavita/ChatDoctor-HealthCareMagic-100k"], split="train")
result = dataset.map(medic_transform, remove_columns=["input", "output", "instruction"])
result.save_to_disk("/usr/Datasets-LLMS/CodeFeedback-Filtered-Instruction-Transformed")

dataset = load_dataset(DATASETS["fnlp/moss-002-sft-data"], split="train")
result = dataset.map(mult_transform, remove_columns=["num_turns", "plain_text", "prefix", "id"])
result.save_to_disk("/usr/Datasets-LLMS/CodeFeedback-Filtered-Instruction-Transformed")