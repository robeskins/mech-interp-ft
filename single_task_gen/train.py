from utils.ft_utils import setup_model_peft, convert_csv_to_json, load_and_preprocess_data
from transformers import (
    TrainingArguments,
    Trainer
)
from pathlib import Path
import gc
import torch
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

def check_run_name(scratch_cache_dir_checkpoints: str, run_name: str):
    run_dir = scratch_cache_dir_checkpoints / run_name  

    if run_dir.exists():
        answer = input(f"The directory {run_dir} already exists. Do you want to delete it? (y/n): ").strip().lower()
        if answer == 'y':
            shutil.rmtree(run_dir)
            print(f"Deleted {run_dir}")
        else:
            print("Exiting script. Please choose a different run_name or delete the directory manually.")
            exit(1)

def main(tasks: list[str],
         dataset_root: str,
         run_name: str,
         model_name: str,
         lora_r: int,
         lora_alpha: int,
         save_steps: int,
         max_steps: int,
         learning_rate: float,
         model_cache_dir: str,
         checkpoint_cache_dir: str,
         per_device_train_batch_size: int = 8,
         gradient_accumulation_steps: int = 5,
         weight_decay: float = 0.01,
         logging_steps: int = 10,
         warmup_steps: int = 0
         ):
    
    checkpoint_cache_dir = Path(checkpoint_cache_dir)
    checkpoint_cache_dir.mkdir(parents=True, exist_ok=True)
    check_run_name(checkpoint_cache_dir, run_name)

    #Iterate through tasks
    for task in tasks:
        checkpoints_dir = checkpoint_cache_dir / run_name / task / "checkpoints"
        training_args_0 = TrainingArguments(output_dir= checkpoints_dir,
                                          per_device_train_batch_size = per_device_train_batch_size,
                                          gradient_accumulation_steps = gradient_accumulation_steps,
                                          weight_decay = weight_decay,
                                          logging_steps = logging_steps,
                                          save_steps=save_steps,
                                          save_strategy="steps",
                                          learning_rate=learning_rate,
                                          max_steps= max_steps,
                                          fp16=True,
                                          report_to="none",
                                          lr_scheduler_type="linear", 
                                          warmup_steps = warmup_steps
                                  )         

        model, tokenizer = setup_model_peft(
            model_name = model_name,
            scratch_cache_dir = model_cache_dir,
            lora_r = lora_r,
            lora_alpha = lora_alpha
        )   

        tokenized_datasets = get_tokenized_datasets(Path(dataset_root) / task  / 'datasets_csv', tokenizer)
    
        trainer = Trainer(
            model=model,
            args=training_args_0,
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
    tasks = ['Add']
    main(tasks = tasks,
         dataset_root = 'task_set_0',
         run_name = "test",
         model_name = "EleutherAI/pythia-1.4b-deduped",
         lora_r = 16,
         lora_alpha = 32,
         save_steps = 27,
         max_steps = 270,
         learning_rate = 3e-4,
         model_cache_dir = "/mnt/faster0/rje41/.cache/huggingface",
         checkpoint_cache_dir = "/mnt/faster0/rje41/checkpoints/experiment_0",
         per_device_train_batch_size = 8,
         gradient_accumulation_steps = 4,
         weight_decay = 0.01,
         logging_steps = 10,
         warmup_steps=0
         )
    
    