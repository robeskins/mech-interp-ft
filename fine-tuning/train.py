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

def load_and_preprocess_data(train_file, validation_file, tokenizer, max_length): 

    data_files = {
        'train': train_file,
        'validation': validation_file
    }
    dataset = load_dataset('json', data_files=data_files)
    
    def preprocess_function(examples):

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

def convert_csv_to_json(train_file_csv: str, test_file_csv: str, output_dir: str) -> None:
    data_train = []
    subfolder = os.path.join(output_dir, f"datasets_jsonl")
    filename_train_jsonl = os.path.join(subfolder, "train.jsonl")
    filename_test_jsonl = os.path.join(subfolder, "validation.jsonl")
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

def setup_model_peft(model_name: str, scratch_cache_dir: str, lora_r: int, lora_alpha: int):
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
        r=lora_r,  
        lora_alpha=lora_alpha,  
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    batch_size, seq_len = labels.shape
    correct = 0
    total = 0

    for i in range(batch_size):
        label_seq = labels[i]
        pred_seq = predictions[i]
        valid_indices = np.where(label_seq != -100)[0]
        if len(valid_indices) == 0:
            continue
        
        first_index = valid_indices[0]
        pred_token_id = pred_seq[first_index-1]
        label_token_id = label_seq[first_index]

        if pred_token_id == label_token_id:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}

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
        tokenizer=tokenizer,
        compute_metrics=compute_accuracy,
    )

    print("\n🔍 Evaluating model before training...")
    metrics = trainer.evaluate()
    print("Pre-training evaluation metrics:", metrics)
    trainer.train()
    
    print("\n🔍 Evaluating model after training...")
    metrics = trainer.evaluate()
    print("Post-training evaluation metrics:", metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        data = yaml.safe_load(file)

    train_adapter(data)