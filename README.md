# CamelSloth

**CamelSloth** is a meta-library that automates the process of:

1. **Generating Chain-of-Thought (CoT) data** from a set of user-provided Q&A pairs (using [CAMEL-AI](https://github.com/camel-ai/camel)).
2. **Fine-tuning** a Large Language Model (LLM) using LoRA with [Unsloth](https://github.com/unslothai/unsloth) for efficient training.
3. **Optionally uploading** the fine-tuned model to [Hugging Face](https://huggingface.co).

This process is controlled via a single **configuration file** (YAML) that the user fills out with their preferences, model choices, API keys, and data paths. Running one command (`camelsloth train --config=...`) orchestrates the entire pipeline end-to-end:

- **QA -> CoT** data generation with CAMEL
- **Prompt-format** conversion into an SFT-ready dataset
- **LoRA Fine-tuning** with Unsloth
- **Model saving** locally
- **(Optional) Pushing** the model/dataset to Hugging Face

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Configuration File](#configuration-file)
5. [Sample Data](#sample-data)
6. [Usage](#usage)
7. [How It Works](#how-it-works)
8. [Tips & Notes](#tips--notes)
9. [License](#license)

---

## Prerequisites

- **Python** 3.8 or higher.
- A GPU environment (e.g., local or Colab) is **strongly recommended** if you want to do real fine-tuning.
- **API Keys**:
  - **OpenAI API Key** (or Qwen or another if you adapt it) for chain-of-thought data generation (optional if you only want to run some local logic).
  - **Hugging Face** token if you want to push your fine-tuned model to the Hugging Face Hub.

---

## Installation

1. Clone this repo:

   ```bash
   git clone https://github.com/YourUsername/CamelSloth.git
   cd CamelSloth
   ```

2. (Optional, but recommended) Create a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies in **editable** mode:

   ```bash
   pip install -e .
   ```

This will install `camelsloth` and its dependencies: `camel-ai`, `unsloth`, `transformers`, `trl`, `huggingface-hub`, `pyyaml`, etc.

---

## Project Structure

```
CamelSloth/
├── camelsloth/
│   ├── __init__.py
│   ├── config_parser.py
│   ├── data_generation.py
│   ├── data_preprocessing.py
│   ├── main.py
│   ├── sft_finetuning.py
│   └── utils.py
├── examples/
│   ├── sample_config.yaml
│   └── qa_data.json
├── setup.py
├── LICENSE
└── README.md  <-- This file
```

The **`camelsloth`** directory contains all the Python modules:

- `main.py` – The CLI entry point.  
- `data_generation.py` – Logic for generating CoT from Q&A using CAMEL.  
- `data_preprocessing.py` – Converting Q&A + CoT into Alpaca or other SFT-friendly prompts.  
- `sft_finetuning.py` – Fine-tuning code using Unsloth’s LoRA approach.  
- `utils.py` – Utilities for saving models, pushing to Hugging Face, etc.

The **`examples`** folder contains a minimal working sample config (`sample_config.yaml`) plus sample QA data (`qa_data.json`).

---

## Configuration File

All user preferences are stored in a YAML file. For example, see [examples/sample_config.yaml](examples/sample_config.yaml).

**Key Sections**:

1. **api_keys**  
   - `openai`: Your OpenAI API key (e.g., `"sk-xxx"`). Or leave empty if you do not want chain-of-thought generation with OpenAI.  
   - `huggingface`: Your HF token (e.g., `"hf_xxx"`). If present, we can push the model/dataset to the Hugging Face Hub.

2. **data_generation**  
   - `qa_data_path`: Path to the user’s Q&A JSON. In the sample, it’s `"examples/qa_data.json"`.  
   - `model_platform`, `model_type`, and `system_message`: For setting up CAMEL’s ChatAgent.  

3. **fine_tuning**  
   - `base_model_name`: The base LLM to fine-tune (e.g. `"unsloth/Qwen2.5-1.5B"`).  
   - `max_seq_length`, `dtype`, `load_in_4bit`, `lora_rank`, etc.: All LoRA details (alpha, dropout, etc.).  

4. **training**  
   - Training hyperparameters (batch size, steps, lr, etc.).  

5. **dataset**  
   - Optional name of a HF dataset.  
   - A prompt template if you want to do an Alpaca-like approach.  

6. **huggingface_upload**  
   - If you want to upload the dataset or model, specify your username and dataset name.  

7. **model_saving**  
   - Where to save the final LoRA adapters or the final model.  
   - Where on HF to push it (`huggingface_repo_name`).  

> **Important:** You **must** fill in or remove the sections relevant to your pipeline. If you skip data generation (no API key), you can supply your own CoT or skip the generation step.

---

## Sample Data

In `examples/qa_data.json`, you’ll see a minimal example:

```json
{
  "What is 2+2?": "4",
  "How many letters are in 'banana'?": "3"
}
```

This is the simplest possible Q&A format:  
```json
{
  "question1": "answer1",
  "question2": "answer2"
}
```

You can expand it to any number of questions. The system will generate chain-of-thought for each question and combine it with the final answer.

---

## Usage

1. **(Optional) Edit** `examples/sample_config.yaml` with your own keys or data. In particular:  
   - `api_keys.openai` – Insert your OpenAI key or comment it out if you want to skip CoT generation.  
   - `api_keys.huggingface` – Insert your token if you want to push the final model.  
   - `data_generation.qa_data_path` – If your Q&A data is somewhere else.  
   - `fine_tuning.base_model_name` – If you prefer a different base model.  
   - Tweak training hyperparameters, etc.

2. **Run** the training pipeline from the project root. For example:

   ```bash
   # If installed via pip install -e .
   python -m camelsloth.main train --config examples/sample_config.yaml
   ```

   or, if you set up an entry point in your `setup.py`, you might do:

   ```bash
   camelsloth train --config examples/sample_config.yaml
   ```
   
   This will:
   - Load `sample_config.yaml`.  
   - Generate chain-of-thought using CAMEL’s CoTDataGenerator for each question in `qa_data.json`.  
   - Transform that data into an Alpaca-like SFT dataset.  
   - Fine-tune your chosen base model using Unsloth’s LoRA approach.  
   - Save the final LoRA adapter to `lora_model/`.  
   - If a huggingface token is present, push that model to your HF repo.

3. **Check** the result in `lora_model/` or your Hugging Face repo.

---

## How It Works

1. **Data Generation** (`camelsloth/data_generation.py`):  
   - Uses `CoTDataGenerator` from CAMEL.  
   - For each question, it queries the LLM to produce a step-by-step reasoning chain plus the final answer.  
   - The chain-of-thought text is stored in a dictionary.

2. **Data Preprocessing** (`data_preprocessing.py`):  
   - Converts the Q + CoT to a supervised fine-tuning style format (e.g. Alpaca style).  
   - Saves a `.jsonl` file which can be loaded by Hugging Face `datasets`.

3. **Fine-tuning** (`sft_finetuning.py`):  
   - Loads the base model with Unsloth’s `FastLanguageModel` plus LoRA adapters.  
   - Creates a training dataset from the `.jsonl` file.  
   - Runs SFT (Supervised Fine-tuning) with `SFTTrainer` from Hugging Face’s TRL library.  
   - Saves the final model or LoRA adapter to a local folder.

4. **Optional Upload** (`utils.py`):  
   - If the user provides a Hugging Face token and a `huggingface_repo_name`, it pushes the final folder to the HF Hub.

---

## Tips & Notes

1. **OpenAI Key**: If you don’t provide one, the data generation step will fail. You can remove or bypass that step by customizing `main.py`, or by providing your own CoT data directly.  
2. **Low-Rank Adaptation**: Because we’re using LoRA, the VRAM usage is significantly reduced. This can run on a **T4** GPU or possibly smaller.  
3. **Model Architecture**: If you want Qwen, Mistral, Llama-3, or any other base model, you must specify the correct `base_model_name`. We have tested `unsloth/Qwen2.5-1.5B` as an example.  
4. **Push to Hugging Face**: If you omit or comment out `huggingface` token in the config, that step is skipped.  
5. **Where does my data go?** The pipeline will create an intermediate `.jsonl` for training in your local folder. You can remove it or keep it for debugging.  
6. **Production or PoC**: This library is provided as a **Proof of Concept**. For production usage, consider adding robust validations, logging, and error handling.  

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to adapt and extend it!

---

### Quick Start Recap

1. **Clone** & `pip install -e .`
2. **Fill in** your config in `examples/sample_config.yaml` (or use defaults).
3. **Run**:
   ```bash
   python -m camelsloth.main train --config examples/sample_config.yaml
   ```
4. **Done**! Find your fine-tuned model in `lora_model/` or on HF.

Happy CoT Generation & Fine-Tuning with **CamelSloth**!