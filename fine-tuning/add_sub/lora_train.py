import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from datasets import Dataset

scratch_cache_dir = "/mnt/fast0/rje41/.cache/huggingface"    
model_path = "EleutherAI/pythia-1.4b-deduped"
os.makedirs(os.path.join(scratch_cache_dir, "hub"), exist_ok=True)
os.makedirs(os.path.join(scratch_cache_dir, "datasets"), exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             cache_dir=os.path.join(scratch_cache_dir, "hub")
                                            )

model.config.use_cache = False 
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
   r=64,
   lora_alpha=128,
   lora_dropout=0.05,
   bias="none",
   task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

file_path = '../../datasets/add_sub.csv'

df = pd.read_csv(file_path)
dataset = Dataset.from_pandas(df)

def concat_clean_label(example):
    example["clean_label"] = example["clean"] + str(example["label"])
    return example

dataset = dataset.map(concat_clean_label)

def create_labels_no_mask(example):
    tokenized = tokenizer(example["clean_label"], truncation=True, padding="max_length", max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(create_labels_no_mask, batched=False, remove_columns=['clean', 'label', 'corrupted', 'clean_label'])

train_valid = dataset.train_test_split(test_size=0.1)
train_valid_split = train_valid['train'].train_test_split(test_size=0.1)

training_args = TrainingArguments(
    output_dir="./results2",
    logging_steps=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    save_strategy="steps", 
    save_steps = 1000,
    max_steps = 10000,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_valid_split['train'],
    # eval_dataset=train_valid_split['test'], 
    tokenizer=tokenizer
    )

trainer.train()