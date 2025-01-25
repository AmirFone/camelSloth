# camelsloth/data_generation.py

import json
import os
from camel.agents import ChatAgent
from camel.datagen.cotdatagen import CoTDataGenerator
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig

def generate_cot_data(config: dict) -> dict:
    """
    Generates chain-of-thought data from Q -> A pairs using CAMEL's CoTDataGenerator.
    """
    # Set up environment variables for openai (or whichever is needed)
    openai_key = config["api_keys"].get("openai", None)
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    
    data_gen_cfg = config["data_generation"]
    model_platform = data_gen_cfg["model_platform"]
    model_type = data_gen_cfg["model_type"]

    # Load QA data
    qa_data_path = data_gen_cfg["qa_data_path"]
    with open(qa_data_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # Create the model
    if model_platform == "OPENAI":
        chosen_platform = ModelPlatformType.OPENAI
    elif model_platform == "QWEN":
        chosen_platform = ModelPlatformType.QWEN
    else:
        chosen_platform = ModelPlatformType.OPENAI

    # For example, we just pick ChatGPTConfig for demonstration
    model_config = ChatGPTConfig().as_dict()

    model = ModelFactory.create(
        model_platform=chosen_platform,
        model_type=ModelType[model_type],
        model_config_dict=model_config,
    )

    # System message for chain-of-thought generation
    sys_msg = data_gen_cfg["system_message"]

    chat_agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        message_window_size=10,
    )

    # Use CoTDataGenerator
    cot_generator = CoTDataGenerator(chat_agent, golden_answers=qa_data)
    generated_answers = {}

    for question in qa_data.keys():
        answer_str = cot_generator.get_answer(question)
        generated_answers[question] = answer_str

    # Optionally verify correctness
    # ...
    return generated_answers
