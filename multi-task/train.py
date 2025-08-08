import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
import numpy as np
from utils.eap_utils import run_metrics
from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data

def train_adapter(dataset_path, model_name, scratch_cache_dir, lora_r, lora_alpha, max_length):
    train_file_csv = Path(dataset_path) / "train.csv"
    test_file_csv = Path(dataset_path) / "validation.csv"
    parent_path = Path(dataset_path).parent

    print(f"Processing: {train_file_csv}")

    model, tokenizer = setup_model_peft(model_name, scratch_cache_dir, lora_r, lora_alpha)

    filename_train_jsonl, filename_test_jsonl = convert_csv_to_json(
        str(train_file_csv), str(test_file_csv), str(parent_path)
    )
    tokenized_datasets = load_and_preprocess_data(str(filename_train_jsonl), str(filename_test_jsonl), tokenizer, max_length)    
    checkpoints_dir = str(Path(parent_path) / "checkpoints")
    training_args = TrainingArguments(output_dir= checkpoints_dir,
                                          per_device_train_batch_size=8,
                                          gradient_accumulation_steps=4,
                                          weight_decay=0.01,
                                          logging_dir="./logs",
                                          logging_steps=10,
                                          save_steps=250,
                                          save_strategy="steps",
                                          learning_rate=3e-4,
                                          max_steps=1000,
                                          fp16=True,
                                          report_to="none",
                                      )
            
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )
    trainer.train()

if __name__ == '__main__':
    dataset_paths = ['tasks/add_sub/datasets_csv', 'tasks/sequence/datasets_csv', 'tasks/MultiTask/add_seq/datasets_csv']
    model_name =  "EleutherAI/pythia-1.4B-deduped"
    scratch_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
    lora_r = 64
    lora_alpha = 128
    max_length = 32
    
    for dataset_path in dataset_paths:
        train_adapter(dataset_path, model_name, scratch_cache_dir, lora_r, lora_alpha, max_length)