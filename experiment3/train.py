from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
from transformers import (
    TrainingArguments,
    Trainer
)
from pathlib import Path
from sklearn.utils import shuffle
import json
import pandas as pd

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
    
    df_combined = pd.concat([df1, df2], ignore_index=True)
    df_shuffled = shuffle(df_combined, random_state=42).reset_index(drop=True)
    
    return df_shuffled

def get_combined_dataset(folder_a: Path, folder_b: Path, half:bool):
    train_path_a = folder_a / 'datasets_csv' / 'train.csv'
    train_path_b = folder_b / 'datasets_csv' / 'train.csv'
    validation_path_a = folder_a / 'datasets_csv' / 'validation.csv'
    validation_path_b = folder_b / 'datasets_csv' / 'validation.csv'

    train_combined_df = get_combined_pd(train_path_a, train_path_b)
    validation_combined_df = get_combined_pd(validation_path_a, validation_path_b, half)

    return train_combined_df, validation_combined_df

def main(folders, folder_combos: list[str]):
    #load first adapter and train
    scratch_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"
    model_name = "EleutherAI/pythia-1.4b-deduped"
    lora_r = 32
    lora_alpha = 64
    save_steps = 72
    max_steps = 360

    folder_combos = [(Path(folders[0]) / task_a, Path(folders[0]) / task_b) for task_a, task_b in folder_combos]
 
    for task_a, task_b in folder_combos:
        task_name = f"{task_a.name}_{task_b.name}"
        task_a_base_dir = Path(f"./results/{task_a.name}") 
        task_b_base_dir = Path(f"./results/{task_b.name}") 
        multi_base_dir = Path(f"./results/{task_name}")    

        #TASK A
        if not task_a_base_dir.exists():
            print('training:',task_a)
            checkpoints_dir = task_a_base_dir / "checkpoints"
            training_args = TrainingArguments(output_dir= checkpoints_dir,
                                              per_device_train_batch_size=8,
                                              gradient_accumulation_steps=5,
                                              weight_decay=0.01,
                                              logging_steps=10,
                                              save_steps=save_steps,
                                              save_strategy="steps",
                                              learning_rate=3e-4,
                                              max_steps=max_steps,
                                              fp16=True,
                                              report_to="none",
                                      )         

            model, tokenizer = setup_model_peft(
                model_name = model_name,
                scratch_cache_dir = scratch_cache_dir,
                lora_r = lora_r,
                lora_alpha = lora_alpha
            )   
            tokenized_datasets = get_tokenized_datasets(task_a / 'datasets_csv', tokenizer)
    
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=tokenizer
            )
            trainer.train() 

            task_info = {
                "task_validation_path": str(task_a),
            }       

            task_a_base_dir.mkdir(parents=True, exist_ok=True)     
            with open(task_a_base_dir / "task_info.json", "w") as f:
                json.dump(task_info, f, indent=2)

        #Task B
        if not task_b_base_dir.exists():
            print('Training:',task_b)
            checkpoints_dir = task_b_base_dir / "checkpoints"
            training_args = TrainingArguments(output_dir= checkpoints_dir,
                                              per_device_train_batch_size=8,
                                              gradient_accumulation_steps=5,
                                              weight_decay=0.01,
                                              logging_steps=10,
                                              save_steps=save_steps,
                                              save_strategy="steps",
                                              learning_rate=3e-4,
                                              max_steps=max_steps,
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
            tokenized_datasets = get_tokenized_datasets(task_b / 'datasets_csv', tokenizer)
    
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=tokenizer
            )
            trainer.train() 

            task_info = {
                "task_validation_path": str(task_b),
            }       

            task_b_base_dir.mkdir(parents=True, exist_ok=True)     
            with open(task_b_base_dir / "task_info.json", "w") as f:
                json.dump(task_info, f, indent=2)


        #TASK A & B
        if not multi_base_dir.exists():
            train_combined_df, validation_combined_df = get_combined_dataset(task_a, task_b, True)
            combined_dataset_path = Path(folders[0]) / task_name / "datasets_csv"
            combined_dataset_path.mkdir(parents=True, exist_ok=True)

            train_combined_df.to_csv(combined_dataset_path / "train.csv", index=False)
            validation_combined_df.to_csv(combined_dataset_path / "validation.csv", index=False)

            checkpoints_dir = multi_base_dir /"checkpoints"
            print('Training:',task_name)
            training_args = TrainingArguments(output_dir= checkpoints_dir,
                                              per_device_train_batch_size=8,
                                              gradient_accumulation_steps=5,
                                              weight_decay=0.01,
                                              logging_steps=10,
                                              save_steps=save_steps,
                                              save_strategy="steps",
                                              learning_rate=3e-4,
                                              max_steps=max_steps,
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
            tokenized_datasets = get_tokenized_datasets(Path(folders[0]) / task_name / 'datasets_csv', tokenizer)
    
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=tokenizer
            )
            trainer.train() 
        
            task_info = {
                "task_validation_path": str(combined_dataset_path.parent),
                "task_a_validation_path": str(task_a),
                "task_b_validation_path": str(task_b),
            }       

            multi_base_dir.mkdir(parents=True, exist_ok=True)     
            with open(multi_base_dir / "task_info.json", "w") as f:
                json.dump(task_info, f, indent=2)

if __name__ == '__main__':
    folder_combos = [('Add','Sub'),
                     ('AddSub','CondAddSub'),
                     ('Sub','Abs')]

    folders = ['tasks']
    main(folders, folder_combos)