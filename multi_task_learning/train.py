from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
from transformers import (
    TrainingArguments,
    Trainer
)
from pathlib import Path
from sklearn.utils import shuffle
import json
import pandas as pd
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


def get_combined_pd(path_1, path_2, half=False):
    df1 = pd.read_csv(path_1)
    df2 = pd.read_csv(path_2)  
    
    if half:
        df1 = df1.sample(n=len(df1) // 2, random_state=42)
        df2 = df2.sample(n=len(df2) // 2, random_state=42)
    
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    df_interleaved = pd.concat([df1, df2]) \
                       .sort_index(kind="merge") \
                       .reset_index(drop=True)
    
    return df_interleaved

def get_combined_dataset(folder_a: Path, folder_b: Path, half:bool):
    train_path_a = folder_a / 'datasets_csv' / 'train.csv'
    train_path_b = folder_b / 'datasets_csv' / 'train.csv'
    validation_path_a = folder_a / 'datasets_csv' / 'validation.csv'
    validation_path_b = folder_b / 'datasets_csv' / 'validation.csv'

    train_combined_df = get_combined_pd(train_path_a, train_path_b)
    validation_combined_df = get_combined_pd(validation_path_a, validation_path_b, half)

    return train_combined_df, validation_combined_df

def train_adapter(checkpoint_root_dir: str,
                  per_device_train_batch_size: int,
                  gradient_accumulation_steps: int,
                  weight_decay: float,
                  logging_steps: int,
                  save_steps: int, 
                  learning_rate: float,
                  max_steps: int,
                  lora_r: int,
                  lora_alpha: int,
                  model_name: str,
                  model_cache_dir: str,
                  task_path: Path,
                  ):
    checkpoints_dir = checkpoint_root_dir / "checkpoints"
    training_args = TrainingArguments(output_dir= checkpoints_dir,
                                      per_device_train_batch_size=per_device_train_batch_size,
                                      gradient_accumulation_steps=gradient_accumulation_steps,
                                      weight_decay=weight_decay,
                                      logging_steps=logging_steps,
                                      save_steps=save_steps,
                                      save_strategy="steps",
                                      learning_rate=learning_rate,
                                      max_steps=max_steps,
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
    tokenized_datasets = get_tokenized_datasets(task_path / 'datasets_csv', tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer
    )
    trainer.train() 

def main(dataset_root_folder: str, 
         task_combos: list[str],
         model_cache_dir: str,
         checkpoints_cache_dir: str,
         run_name: str,
         model_name: str,
         lora_r: int,
         lora_alpha: int,
         save_steps: int,
         max_steps: int,
         per_device_train_batch_size: int,
         gradient_accumulation_steps: int,
         logging_steps: int,
         learning_rate: float,
         weight_decay: float,
         half_dataset: bool):
    #load first adapter and train
    model_cache_dir = Path(model_cache_dir)
    checkpoints_cache_dir = Path(checkpoints_cache_dir)
    checkpoints_cache_dir.mkdir(parents=True, exist_ok=True)
    
    run_dir = checkpoints_cache_dir / run_name  

    if run_dir.exists():
        answer = input(f"The directory {run_dir} already exists. Do you want to delete it? (y/n): ").strip().lower()
        if answer == 'y':
            shutil.rmtree(run_dir)
            print(f"Deleted {run_dir}")
        else:
            print("Exiting script. Please choose a different run_name or delete the directory manually.")
            exit(1)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    folder_combos = [(Path(dataset_root_folder) / task_a, Path(dataset_root_folder) / task_b) for task_a, task_b in task_combos]
 
    for task_a, task_b in folder_combos:
        task_name = f"{task_a.name}_{task_b.name}"
        task_a_base_dir = checkpoints_cache_dir / run_name / task_a.name 
        task_b_base_dir = checkpoints_cache_dir / run_name / task_b.name 
        multi_base_dir = checkpoints_cache_dir / run_name / task_name

        #TASK A
        if not task_a_base_dir.exists():
            print('training:',task_a)    

            train_adapter(checkpoint_root_dir = task_a_base_dir,
                          per_device_train_batch_size = per_device_train_batch_size,
                          gradient_accumulation_steps = gradient_accumulation_steps,
                          weight_decay = weight_decay,
                          logging_steps = logging_steps,
                          save_steps = save_steps, 
                          learning_rate = learning_rate,
                          max_steps = max_steps,
                          lora_r = lora_r,
                          lora_alpha = lora_alpha,
                          model_name = model_name,
                          model_cache_dir = model_cache_dir,
                          task_path = task_a,
                          )
            task_info = {
                "task_validation_path": str(task_a),
            }       
            task_a_base_dir.mkdir(parents=True, exist_ok=True) 
            with open(task_a_base_dir / "task_info.json", "w") as f:
                json.dump(task_info, f, indent=2)
            
        #Task B
        if not task_b_base_dir.exists():
            print('training:',task_b)
            
            train_adapter(checkpoint_root_dir = task_b_base_dir,
                          per_device_train_batch_size = per_device_train_batch_size,
                          gradient_accumulation_steps = gradient_accumulation_steps,
                          weight_decay = weight_decay,
                          logging_steps = logging_steps,
                          save_steps = save_steps, 
                          learning_rate = learning_rate,
                          max_steps = max_steps,
                          lora_r = lora_r,
                          lora_alpha = lora_alpha,
                          model_name = model_name,
                          model_cache_dir = model_cache_dir,
                          task_path = task_b
                          )
            task_info = {
                "task_validation_path": str(task_b),
            }       
            task_b_base_dir.mkdir(parents=True, exist_ok=True) 
            with open(task_b_base_dir / "task_info.json", "w") as f:
                json.dump(task_info, f, indent=2)

        #TASK A & B
        if not multi_base_dir.exists():
            train_combined_df, validation_combined_df = get_combined_dataset(task_a, task_b, half_dataset)
            combined_dataset_path = Path(dataset_root_folder) / task_name / "datasets_csv"
            combined_dataset_path.mkdir(parents=True, exist_ok=True)

            train_combined_df.to_csv(combined_dataset_path / "train.csv", index=False)
            validation_combined_df.to_csv(combined_dataset_path / "validation.csv", index=False)
            
            print('Training:',task_name)
            train_adapter(checkpoint_root_dir = multi_base_dir,
                          per_device_train_batch_size = per_device_train_batch_size,
                          gradient_accumulation_steps = gradient_accumulation_steps,
                          weight_decay = weight_decay,
                          logging_steps = logging_steps,
                          save_steps = save_steps, 
                          learning_rate = learning_rate,
                          max_steps = max_steps,
                          lora_r = lora_r,
                          lora_alpha = lora_alpha,
                          model_name = model_name,
                          model_cache_dir = model_cache_dir,
                          task_path = Path(dataset_root_folder) / task_name
                          )
        
            task_info = {
                "task_validation_path": str(combined_dataset_path.parent),
                "task_a_validation_path": str(task_a),
                "task_b_validation_path": str(task_b),
            }       
            
            multi_base_dir.mkdir(parents=True, exist_ok=True) 
            with open(multi_base_dir / "task_info.json", "w") as f:
                json.dump(task_info, f, indent=2)

if __name__ == '__main__':
    multi_task_pairs = [
        ("Add", "Sub"),
        ('Sub','Abs'),
        ('Modulo', 'Add'),
        ('FloorDiv','Modulo')
    ]

    main(dataset_root_folder = 'task_set_1', 
         task_combos = multi_task_pairs,
         model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
         checkpoints_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_3",
         run_name = 'task_set_1_half_true',
         model_name = "EleutherAI/pythia-1.4b-deduped",
         lora_r = 4,
         lora_alpha = 8,
         save_steps = 18,
         max_steps = 180,
         per_device_train_batch_size = 8,
         gradient_accumulation_steps = 5,
         logging_steps = 10,
         learning_rate = 3e-4,
         weight_decay = 0.01,
         half_dataset = True)
    
    main(dataset_root_folder = 'task_set_1', 
         task_combos = multi_task_pairs,
         model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
         checkpoints_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_3",
         run_name = 'task_set_1_half_false',
         model_name = "EleutherAI/pythia-1.4b-deduped",
         lora_r = 4,
         lora_alpha = 8,
         save_steps = 18,
         max_steps = 180,
         per_device_train_batch_size = 8,
         gradient_accumulation_steps = 5,
         logging_steps = 10,
         learning_rate = 3e-4,
         weight_decay = 0.01,
         half_dataset = False)