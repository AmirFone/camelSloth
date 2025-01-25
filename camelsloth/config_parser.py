# camelsloth/config_parser.py

import yaml
import os

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file and returns it as a Python dict.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
