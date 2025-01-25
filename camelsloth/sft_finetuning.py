# camelsloth/sft_finetuning.py

import os
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
import json

def finetune_model(config: dict, data_file: str):
    # Load config
    fine_tuning_cfg = config["fine_tuning"]
    training_cfg = config["training"]

    # Prepare base model with Unsloth
    base_model_name = fine_tuning_cfg["base_model_name"]
    max_seq_length = fine_tuning_cfg["max_seq_length"]
    dtype = fine_tuning_cfg["dtype"]
    load_in_4bit = fine_tuning_cfg["load_in_4bit"]
    out_dir = fine_tuning_cfg["output_dir"]

    # Example: Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Setup LoRA
    lora_rank = fine_tuning_cfg["lora_rank"]
    target_modules = fine_tuning_cfg["lora_target_modules"]
    lora_alpha = fine_tuning_cfg["lora_alpha"]
    lora_dropout = fine_tuning_cfg["lora_dropout"]
    bias = fine_tuning_cfg["bias"]
    random_state = fine_tuning_cfg["random_state"]
    use_gradient_checkpointing = fine_tuning_cfg["use_gradient_checkpointing"]
    use_rslora = fine_tuning_cfg["use_rslora"]
    loftq_config = fine_tuning_cfg["loftq_config"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=loftq_config,
    )

    # Load data as a HF dataset
    raw_data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    dataset = Dataset.from_list(raw_data)

    # We assume there's a 'text' field after some prompt formatting or we do that now
    # Let's do a naive approach
    def formatting_func(example):
        instruction = example["instruction"]
        input_str = example["input"]
        output_str = example["output"]
        # Insert your prompt from config if needed
        # For demonstration we just do:
        text = f"Below is an instruction...\nInstruction: {instruction}\nInput: {input_str}\nResponse: {output_str}"
        return {"text": text}

    dataset = dataset.map(formatting_func)

    # Build training arguments
    trainer_args = TrainingArguments(
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        warmup_steps=training_cfg["warmup_steps"],
        max_steps=training_cfg["max_steps"],
        learning_rate=training_cfg["learning_rate"],
        fp16=training_cfg["fp16"] and not is_bfloat16_supported(),
        bf16=training_cfg["bf16"] and is_bfloat16_supported(),
        logging_steps=training_cfg["logging_steps"],
        optim=training_cfg["optim"],
        weight_decay=training_cfg["weight_decay"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        seed=training_cfg["seed"],
        output_dir=out_dir,
        report_to=training_cfg["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=trainer_args,
    )

    trainer.train()

    # Return trainer, and the final model
    return trainer
