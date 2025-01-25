# camelsloth/main.py

import argparse
import os
import json
from .config_parser import load_config
from .data_generation import generate_cot_data
from .data_preprocessing import transform_to_alpaca, save_to_jsonl
from .sft_finetuning import finetune_model
from .utils import push_to_huggingface

def run_train(config_path: str):
    """
    Orchestrates the entire pipeline:
    1. Load config
    2. Generate chain-of-thought from QA data
    3. Transform data to SFT style
    4. Fine-tune the model with Unsloth
    5. Optionally push to hugging face
    """
    config = load_config(config_path)

    # 1. Generate CoT
    generated_cot = generate_cot_data(config)
    # 2. Transform to Alpaca style
    alpaca_data = transform_to_alpaca(generated_cot, config)
    
    # Save to JSONL
    sft_data_path = "sft_data.jsonl"
    save_to_jsonl(alpaca_data, sft_data_path)

    # 3. Fine-tune
    trainer = finetune_model(config, sft_data_path)
    trainer.save_model(config["model_saving"]["local_save_path"])
    
    # 4. Hugging Face push (optional)
    if config.get("api_keys", {}).get("huggingface", None):
        # push to HF if token is provided
        push_to_huggingface(config, config["model_saving"]["local_save_path"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train"], help="Which command to run.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()

    if args.command == "train":
        run_train(args.config)
