import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from safetensors.torch import save_file
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
from pathlib import Path
import csv

def load_and_preprocess_data(train_file, validation_file, tokenizer): 

    data_files = {
        'train': train_file,
        'validation': validation_file
    }
    dataset = load_dataset('json', data_files=data_files)
    
    def preprocess_function(examples):
        max_length = 32

        inputs = examples['input']
        outputs = [str(o) for o in examples['output']]

        prompts = [f"{inp}\n" for inp in inputs]
        full_texts = [prompt + out for prompt, out in zip(prompts, outputs)]

        tokenized_full = tokenizer(full_texts, truncation=True, padding='max_length', max_length=max_length)

        tokenized_prompt = tokenizer(prompts, truncation=True, padding='max_length', max_length=max_length)

        labels = []
        for i in range(len(full_texts)):

            prompt_len = len(tokenizer.encode(prompts[i], truncation=True, max_length=max_length))
    
            label = [-100] * prompt_len + tokenized_full['input_ids'][i][prompt_len:]
       
            label = label[:max_length]
      
            if len(label) < max_length:
                label += [-100] * (max_length - len(label))
            labels.append(label)


        tokenized_full['labels'] = labels

        return tokenized_full
    

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
  
    tokenized_datasets = tokenized_datasets.remove_columns(['input', 'output', 'instruction'])
    return tokenized_datasets

def convert_csv_to_json(train_file_csv: str, test_file_csv: str, output_dir: str, id) -> None:
    data_train = []
    subfolder = os.path.join(output_dir, f"prompts_id_{id}")
    filename_train_jsonl = os.path.join(subfolder, "train.jsonl")
    filename_test_jsonl = os.path.join(subfolder, "test.jsonl")
    os.makedirs(subfolder, exist_ok=True)

    data_train = []
    with open(train_file_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data_train.append({"instruction": "", "input": row[0], "output": row[2]})

    with open(filename_train_jsonl, 'w', encoding='utf-8') as f:
        for item in data_train:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

    data_test = []
    with open(test_file_csv, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data_test.append({"instruction": "", "input": row[0], "output": row[2]})

    with open(filename_test_jsonl, 'w', encoding='utf-8') as f:
        for item in data_test:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

    return filename_train_jsonl, filename_test_jsonl

def setup_model_peft(model_name: str, scratch_cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              cache_dir=os.path.join(scratch_cache_dir, "hub"))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   
    
    os.makedirs(os.path.join(scratch_cache_dir, "hub"), exist_ok=True)
    os.makedirs(os.path.join(scratch_cache_dir, "datasets"), exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir=os.path.join(scratch_cache_dir, "hub")
                                                )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,   
        inference_mode=False,          
        r=32,  
        lora_alpha=64,  
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer

MODEL_NAME = 'EleutherAI/pythia-1.4b-deduped'
scratch_cache_dir = "/mnt/faster0/rje41/.cache/huggingface"   
per_device_train_batch_size = 8
gradient_accumulation_steps = 4   

folder_paths = [p for p in Path('datasets_csv').rglob('*') if p.is_dir()]
for folder_path in folder_paths:
    train_file_csv = str(folder_path / 'train.csv')
    test_file_csv = str(folder_path / 'test.csv')
    id = folder_path.name.replace('prompts_id_','')
    print(train_file_csv)

    model, tokenizer = setup_model_peft(model_name = MODEL_NAME,
                            scratch_cache_dir = scratch_cache_dir)
    

    filename_train_jsonl, filename_test_jsonl = convert_csv_to_json(train_file_csv, test_file_csv, 'datasets_json', id)
    
    tokenized_datasets = load_and_preprocess_data(str(filename_train_jsonl), str(filename_test_jsonl), tokenizer)
    
    training_args = TrainingArguments(
                                        output_dir=f'./checkpoints/prompt_template_{id}/',
                                        per_device_train_batch_size=per_device_train_batch_size,
                                        weight_decay=0.01,
                                        logging_dir='./logs',
                                        logging_steps=10,
                                        save_steps=10,
                                        save_strategy="steps",
                                        eval_strategy="steps",   
                                        eval_steps=200,                
                                        fp16=True,
                                        gradient_accumulation_steps=gradient_accumulation_steps,
                                        report_to="none",
                                        learning_rate=3e-4,
                                        max_steps=300,
                                    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer
    )
    trainer.train()