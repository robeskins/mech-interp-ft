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
import csv
import numpy as np
import yaml
import argparse
from utils.eap_utils import run_metrics
from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data

def train_adapter(args: dict):
    dataset_path = Path(args["dataset_path"])
    model_name = args["model_name"]
    scratch_cache_dir = args["scratch_cache_dir"]
    lora_r = args["lora_config"]["r"]
    lora_alpha = args["lora_config"]["alpha"]
    training_args_dict = args["training_args"]
    max_length = args["max_length_strings"]

    train_file_csv = dataset_path / "train.csv"
    test_file_csv = dataset_path / "validation.csv"
    parent_path = dataset_path.parent

    print(f"Processing: {train_file_csv}")

    model, tokenizer = setup_model_peft(model_name, scratch_cache_dir, lora_r, lora_alpha)

    filename_train_jsonl, filename_test_jsonl = convert_csv_to_json(
        str(train_file_csv), str(test_file_csv), str(parent_path)
    )
    tokenized_datasets = load_and_preprocess_data(str(filename_train_jsonl), str(filename_test_jsonl), tokenizer, max_length)

    training_args_dict["output_dir"] = str(parent_path / "checkpoints")

    lr = training_args_dict.get("learning_rate")
    if isinstance(lr, str):
        training_args_dict["learning_rate"] = float(lr)

    training_args = TrainingArguments(**training_args_dict)
    tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(10))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        data = yaml.safe_load(file)

    train_adapter(data)