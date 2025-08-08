import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import csv

def load_and_preprocess_data(train_file, validation_file, tokenizer, max_length = 32, method='simple'):
    """
    method: 'simple' uses the second version with tokenizer.encode for prompt length
            'full' uses the first version that checks attention mask per token
    """
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
            if method == 'full':
                input_ids = tokenized_full['input_ids'][i]
                attention_mask = tokenized_full['attention_mask'][i]
                prompt_len = sum(tokenized_prompt['attention_mask'][i])
                label = [-100] * prompt_len
                for j in range(prompt_len, max_length):
                    if attention_mask[j] == 1:
                        label.append(input_ids[j])
                    else:
                        label.append(-100)
                labels.append(label)

            elif method == 'simple':
                prompt_len = len(tokenizer.encode(prompts[i], truncation=True, max_length=max_length))
                label = [-100] * prompt_len + tokenized_full['input_ids'][i][prompt_len:]
                label = label[:max_length]
                if len(label) < max_length:
                    label += [-100] * (max_length - len(label))
                labels.append(label)

            else:
                raise ValueError(f"Unknown method: {method}")

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


def setup_model_peft(
    model_name: str,
    scratch_cache_dir: str,
    lora_r: int,
    lora_alpha: int,
    load_checkpoint: bool = False,
    latest_checkpoint_path: str = None
):
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=os.path.join(scratch_cache_dir, "hub")
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(os.path.join(scratch_cache_dir, "hub"), exist_ok=True)
    os.makedirs(os.path.join(scratch_cache_dir, "datasets"), exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=os.path.join(scratch_cache_dir, "hub")
    )

    if load_checkpoint:
        if latest_checkpoint_path is None:
            raise ValueError("latest_checkpoint_path must be provided when load_checkpoint=True")

        model = PeftModel.from_pretrained(
            base_model,
            latest_checkpoint_path,
            is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer

