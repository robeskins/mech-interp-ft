from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
from transformers import (
    TrainingArguments,
    Trainer
)
import os
from pathlib import Path
from itertools import combinations, product
from itertools import permutations
import gc
import torch
import json
import shutil

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

def main(task_combos: list[tuple],
         task_root_folder: str,
         model_cache_dir: str,
         checkpoint_cache_dir: str,
         run_name: str, 
         learning_rate_continue: bool,
         max_steps_a: int,
         max_steps_b: int,
         save_steps_task_b: int,
         learning_rate: float,
         lora_r: int,
         lora_alpha: int,
         model_name: str,
         per_device_train_batch_size: int,
         gradient_accumulation_steps: int,
         weight_decay: float,
         logging_steps: int
         ):
    checkpoint_cache_dir = Path(checkpoint_cache_dir)
    checkpoint_cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir = checkpoint_cache_dir / run_name  

    if run_dir.exists():
        answer = input(f"The directory {run_dir} already exists. Do you want to delete it? (y/n): ").strip().lower()
        if answer == 'y':
            shutil.rmtree(run_dir)
            print(f"Deleted {run_dir}")
        else:
            print("Exiting script. Please choose a different run_name or delete the directory manually.")
            exit(1)
    
    task_combos = [(Path(task_root_folder) / task_a, Path(task_root_folder) / task_b) for task_a, task_b in task_combos]
 
    for task_a, task_b in task_combos:

        task_name = f"{task_a.name}->{task_b.name}"
        print('Processing:',task_name)
        
        base_dir = checkpoint_cache_dir / run_name / task_name 
        if base_dir.exists():
            print('base_dir exists')
        base_dir.mkdir(parents=True, exist_ok=True) 

        task_info = {
            "task_a": str(task_a),
            "task_b": str(task_b),
            "task_name": task_name,
            "results_dir": str(base_dir)
        }       

        with open(base_dir / "task_info.json", "w") as f:
            json.dump(task_info, f, indent=2)
        
        checkpoints_dir = base_dir / "checkpoints"

        #Set up first task
        training_args_0 = TrainingArguments(output_dir= checkpoints_dir,
                                          per_device_train_batch_size=per_device_train_batch_size,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          weight_decay=weight_decay,
                                          logging_steps=logging_steps,
                                          save_steps=max_steps_a,
                                          save_strategy="steps",
                                          learning_rate=learning_rate,
                                          max_steps= max_steps_a,
                                          fp16=True,
                                          report_to="none",
                                          lr_scheduler_type="linear", 
                                  )         

        model, tokenizer = setup_model_peft(
            model_name = model_name,
            scratch_cache_dir = model_cache_dir,
            lora_r = lora_r,
            lora_alpha = lora_alpha
        )   
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
            new_lr = learning_rate / 10
        else: 
            new_lr = learning_rate

        del trainer
        del model
        del tokenizer
        del tokenized_datasets
        gc.collect()
        torch.cuda.empty_cache()

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
            scratch_cache_dir = model_cache_dir,
            lora_r = lora_r,
            lora_alpha = lora_alpha,
            load_checkpoint = True,
            latest_checkpoint_path=new_path
        )
        tokenized_datasets = get_tokenized_datasets(task_b / 'datasets_csv', tokenizer)
        training_args_1 = TrainingArguments(output_dir= checkpoints_dir,
                                          per_device_train_batch_size=per_device_train_batch_size,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          weight_decay=weight_decay,
                                          logging_steps=logging_steps,
                                          save_steps=save_steps_task_b,
                                          save_strategy="steps",
                                          learning_rate=new_lr,
                                          max_steps= max_steps_b,
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
    combos = [('AddSub','AddSubAlias')]      #combinations can be manually defined or all permuations

    # tasks = ['Abs','Add','AddSub','AddSubAlias','CondAddSub','FloorDiv','Modulo','Sub']
    # combos = list(permutations(tasks, 2))
    
    main(task_combos = combos,
         task_root_folder = 'tasks',
         model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface", #Change to local cache
         checkpoint_cache_dir = "./checkpoints",
         run_name = 'sequnetial_ft_false', 
         learning_rate_continue = False,
         max_steps_a = 270,
         max_steps_b = 270,
         save_steps_task_b = 270,
         learning_rate = 3e-4,
         lora_r = 32,
         lora_alpha = 64,
         model_name = "EleutherAI/pythia-1.4b-deduped",
         per_device_train_batch_size = 8,
         gradient_accumulation_steps = 5,
         weight_decay = 0.01,
         logging_steps = 10
        )
    
