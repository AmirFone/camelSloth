# camelsloth/utils.py

import os
from huggingface_hub import HfApi, HfFolder

def push_to_huggingface(config: dict, local_path: str):
    api = HfApi()
    hf_token = config["api_keys"]["huggingface"]
    repo_name = config["model_saving"]["huggingface_repo_name"]

    api.upload_folder(
        repo_id=repo_name,
        folder_path=local_path,
        token=hf_token,
        repo_type="model",
        path_in_repo=".",
    )
    print(f"Model pushed to https://huggingface.co/{repo_name}")
