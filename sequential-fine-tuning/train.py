import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data

from transformers import (
    TrainingArguments,
    Trainer
)
from pathlib import Path
import numpy as np

def get_tokenized_datasets(dataset_path: str, tokenizer):
    train_file = Path(dataset_path) / 'train.csv'
    val_file = Path(dataset_path) / 'validation.csv'
    parent_path = Path(dataset_path).parent

    filename_train_jsonl, filename_test_jsonl = convert_csv_to_json(
        str(train_file), str(val_file), str(parent_path)
    )
    tokenized_datasets = load_and_preprocess_data(filename_train_jsonl, 
                                                  filename_test_jsonl, 
                                                  tokenizer,
                                                  max_length = 64)
    return tokenized_datasets

datasets = ['../datasets/pythia-1.4B/three-digit/AddBase8', '../datasets/pythia-1.4B/three-digit/AddBase9','../datasets/pythia-1.4B/three-digit/AddBase10', '../datasets/pythia-1.4B/three-digit/AddBase16',
            '../datasets/pythia-1.4B/two-digit/AddSub', '../datasets/pythia-1.4B/two-digit/AddSubInv', '../datasets/pythia-1.4B/two-digit/CondAddSub',]

model_name = 'EleutherAI/pythia-1.4b-deduped'
lora_r = 32
lora_alpha = 64
scratch_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
max_length = 32

for dataset in datasets:
    dataset_path = Path(dataset)
    parent_path = dataset_path.parent
    end = Path(dataset).name

    model, tokenizer = setup_model_peft(
            model_name = model_name,
            scratch_cache_dir = scratch_cache_dir,
            lora_r = lora_r,
            lora_alpha = lora_alpha
        )   
    tokenized_datasets = get_tokenized_datasets(dataset_path / 'datasets_csv', tokenizer)
    output_dir = str(Path(end) / "checkpoints")
    training_args = TrainingArguments(output_dir= output_dir,
                                      per_device_train_batch_size=8,
                                      gradient_accumulation_steps=4,
                                      weight_decay=0.01,
                                      logging_steps=10,
                                      save_steps=112,
                                      save_strategy="steps",
                                      learning_rate=3e-4,
                                      max_steps=560,
                                      fp16=True,
                                      report_to="none",
                                      lr_scheduler_type="linear", 
                              )     
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )
    trainer.train()
