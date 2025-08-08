from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
from transformers import (
    TrainingArguments,
    Trainer
)
import os
from pathlib import Path
from itertools import combinations, product
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
                                                  max_length = 32)
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

def main(folders, folder_combos: list[str] = None, learning_rate_continue: bool = False):
    #load first adapter and train
    scratch_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
    model_name = "EleutherAI/pythia-1.4b-deduped"
    lora_r = 32
    lora_alpha = 64
    save_steps_task_b = 72
    max_steps = 360
    learning_rate = 3e-4
    
    if not folder_combos:
        folder_combos = get_combos(folders)
    else:
        folder_combos = [(Path(folders[0]) / task_a, Path(folders[0]) / task_b) for task_a, task_b in folder_combos]

    print(len(folder_combos))
 
    for task_a, task_b in folder_combos:
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
                                          gradient_accumulation_steps=5,
                                          weight_decay=0.01,
                                          logging_steps=10,
                                          save_steps=max_steps,
                                          save_strategy="steps",
                                          learning_rate=learning_rate,
                                          max_steps= max_steps,
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
        if learning_rate_continue:
            new_lr = trainer.optimizer.param_groups[0]["lr"]
        else: 
            new_lr = learning_rate

        #Rename final checkpoint to start
        output_dir = training_args_0.output_dir
        checkpoint_folders = [f for f in os.listdir(output_dir) if f.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, f))]       

        if len(checkpoint_folders) == 0:
            raise FileNotFoundError(f"No checkpoint folder found in {output_dir}")
        elif len(checkpoint_folders) > 1:
            raise RuntimeError(f"More than one checkpoint folder found: {checkpoint_folders}")
        else:
            old_path = os.path.join(output_dir, checkpoint_folders[0])
            new_path = os.path.join(output_dir, 'checkpoint-0')
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")

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
        training_args_1 = TrainingArguments(output_dir= checkpoints_dir,
                                          per_device_train_batch_size=8,
                                          gradient_accumulation_steps=5,
                                          weight_decay=0.01,
                                          logging_steps=10,
                                          save_steps=save_steps_task_b,
                                          save_strategy="steps",
                                          learning_rate=new_lr,
                                          max_steps=max_steps,
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
    combos = [
        ('AddSub', 'AddSubAlias'),
        ('AddSub', 'CondAddSub'),
        ('AddSub','AddSub'),
        ('Add','Sub'),
        ('Sub', 'Abs'),
        ('Add', 'Abs'),
        ('Add', 'AddSub'),
        ('Sub', 'AddSub'),
        ('Add', 'CondAddSub'),
        ('Modulo', 'AddSub'),
        ('FloorDiv','Abs'),
        ('Abs', 'Modulo'),
        ('Sub','Modulo'),
        ('Add','Modulo'),
        ('AddSub','Modulo')
    ]
    folders = ['tasks']
    main(folders,folder_combos=combos)
