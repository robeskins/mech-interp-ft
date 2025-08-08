from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
from transformers import (
    TrainingArguments,
    Trainer
)
import os
from pathlib import Path
import argparse
import yaml
from itertools import combinations, product
import random
import gc
import torch

import json

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


def get_combos(task_type_folders: list[str]):
    all_task_folders = []   

    for task_type_folder in task_type_folders:
        subfolders = [p for p in Path(task_type_folder).glob('*') if p.is_dir()]
        all_task_folders.extend(subfolders)

    folder_combos = [
    (a, b) for a, b in product(all_task_folders, repeat=2)
    if a != b
    ]
    print(folder_combos)
    return folder_combos

def main(folders):
    #load first adapter and train
    scratch_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
    model_name = "EleutherAI/pythia-1.4b-deduped"
    lora_r = 32
    lora_alpha = 64

    folder_combos = get_combos(folders)
    # limited_folder_combos = random.sample(folder_combos)
    limited_folder_combos = folder_combos
    print(len(limited_folder_combos))

    for task_a, task_b in limited_folder_combos:
        task_name = f"{task_a.name}->{task_b.name}"
        print(task_name)
        base_dir = Path(f"./results/{task_name}")       

        task_info = {
            "task_a": str(task_a),
            "task_b": str(task_b),
            "task_name": task_name,
            "results_dir": str(base_dir)
        }       

        base_dir.mkdir(parents=True, exist_ok=True)     
        with open(base_dir / "task_info.json", "w") as f:
            json.dump(task_info, f, indent=2)
        
        checkpoints_dir = base_dir / "checkpoints"

        training_args_0 = TrainingArguments(output_dir= checkpoints_dir,
                                          per_device_train_batch_size=8,
                                          gradient_accumulation_steps=4,
                                          weight_decay=0.01,
                                          logging_dir="./logs",
                                          logging_steps=10,
                                          save_steps=500,
                                          save_strategy="steps",
                                          learning_rate=3e-4,
                                          max_steps=500,
                                          fp16=True,
                                          report_to="none",
                                          lr_scheduler_type="linear", 
                                      )     

        model, tokenizer = setup_model_peft(
            model_name = model_name,
            scratch_cache_dir = scratch_cache_dir,
            lora_r = lora_r,
            lora_alpha = lora_alpha
        )   
        print(task_a)
        tokenized_datasets = get_tokenized_datasets(task_a / 'datasets_csv', tokenizer)
    
        trainer = Trainer(
            model=model,
            args=training_args_0,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer
        )
        trainer.train() 

        #Rename final checkpoint to start
        old_path = os.path.join(training_args_0.output_dir, 'checkpoint-500')
        new_path = os.path.join(training_args_0.output_dir, 'checkpoint-0') 

        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")  

        #Load second adapter and train
        model, tokenizer = setup_model_peft(
            model_name = model_name,
            scratch_cache_dir = scratch_cache_dir,
            lora_r = lora_r,
            lora_alpha = lora_alpha,
            load_checkpoint = True,
            latest_checkpoint_path=new_path
        )
        tokenized_datasets = get_tokenized_datasets(task_b / 'datasets_csv', tokenizer)
        training_args_1 = TrainingArguments(
                                            output_dir=checkpoints_dir,
                                            per_device_train_batch_size=8,
                                            gradient_accumulation_steps=4,
                                            weight_decay=0.01,
                                            logging_dir="./logs",
                                            logging_steps=10,
                                            save_steps=100,
                                            save_strategy="steps",
                                            learning_rate=3e-4,
                                            max_steps=500,
                                            fp16=True,
                                            report_to="none",
                                            lr_scheduler_type="linear",
                                        )   

        trainer = Trainer(
            model=model,
            args=training_args_1,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer
        )
        trainer.train()

        del trainer
        del model
        del tokenizer
        del tokenized_datasets
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    folders = ['../dataset_gen/two-digit-arithmetic']
    main(folders)
