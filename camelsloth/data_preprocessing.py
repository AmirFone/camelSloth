# camelsloth/data_preprocessing.py

import json
from datetime import datetime

def transform_to_alpaca(input_dict: dict, config: dict) -> list:
    """
    Transforms the Q + CoT -> A into Alpaca style for SFT.
    input_dict: {question -> chain_of_thought_and_answer}
    """
    ds_config = config["dataset"]
    alpaca_template = ds_config["alpaca_prompt"]

    # We assume each question is an 'instruction' and the chain_of_thought_and_answer is 'output'
    transformed_data = []
    for question, chain_of_thought_output in input_dict.items():
        transformed_data.append({
            "instruction": question,
            "input": "",
            "output": chain_of_thought_output
        })
    return transformed_data

def save_to_jsonl(data_list: list, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
