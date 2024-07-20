from src.utils.prompts import *
import re

def code_feedback_transform(example):
    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT_CODING % example["lang"]
        },
        {
            "role": "user",
            "content": example["query"]
        },
        {
            "role": "assistant",
            "content": example["answer"]
        }
    ]

    return {"text": messages}

def maths_transform(example):
    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT_MATHS
        },
        {
            "role": "user",
            "content": example["instruction"]
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    return {"text": messages}

def medic_transform(example):
    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT_MEDIC
        },
        {
            "role": "user",
            "content": example["input"]
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    return {"text": messages}

def mult_transform(example):
    convos = re.split(r"<eoh>|<eoa>", example["plain_text"])
    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT_MULT
        }
    ]

    num_turns = 0

    for convo in convos:
        if not convo: continue

        if(convo.strip().startswith("[Human]:")):
            messages.append({
                "role": "user",
                "content": convo.replace("[Human]:", "").strip()
            })
            num_turns += 1
        
        if(convo.strip().startswith("[MOSS]:")):
            messages.append({
                "role": "assistant",
                "content": convo.replace("[MOSS]:", "").strip()
            })

    assert(num_turns == example["num_turns"])
    return {"text": messages}


def code_feedback_transform_phi3(example):
    sys_msg = SYS_PROMPT_CODING % example["lang"]
    messages = [
        {
            "role": "user",
            "content": f"{sys_msg}\n" + example["query"]
        },
        {
            "role": "assistant",
            "content": example["answer"]
        }
    ]

    return {"text": messages}

def maths_transform_phi3(example):
    sys_msg = SYS_PROMPT_MATHS
    messages = [
        {
            "role": "user",
            "content": f"{sys_msg}\n" + example["instruction"]
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    return {"text": messages}

def medic_transform_phi3(example):
    sys_msg = SYS_PROMPT_MEDIC
    messages = [
        {
            "role": "user",
            "content": f"{sys_msg}\n" + example["input"]
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    return {"text": messages}

def mult_transform_phi3(example):
    convos = re.split(r"<eoh>|<eoa>", example["plain_text"])
    messages = []

    num_turns = 0

    for convo in convos:
        if not convo: continue

        if(convo.strip().startswith("[Human]:")):
            messages.append({
                "role": "user",
                "content": convo.replace("[Human]:", "").strip()
            })
            num_turns += 1
        
        if(convo.strip().startswith("[MOSS]:")):
            messages.append({
                "role": "assistant",
                "content": convo.replace("[MOSS]:", "").strip()
            })

    assert(num_turns == example["num_turns"])
    return {"text": messages}