from transformers import AutoTokenizer
import csv
from pathlib import Path
import os
import random

def is_single_token(text: str, tokenizer: AutoTokenizer):
    tokens = tokenizer.tokenize(str(text))
    return len(tokens) == 1

def compare_token_count(text_1: str, text_2: str, tokenizer: AutoTokenizer):
    tokens_1 = tokenizer.tokenize(str(text_1))
    tokens_2 = tokenizer.tokenize(str(text_2))
    return len(tokens_1) == len(tokens_2)

def check_uniqueness(prompts: list[list[str]]):
    clean_prompts = [prompt[0] for prompt in prompts]
    return len(set(clean_prompts)) == len(clean_prompts)

def write_to_csv(filename: str, data: list[list[str]]):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clean','corrupted','answer'])

        for row in data:
            writer.writerow(row)

def generate_csvs(prompts: list[list[str]],
                  split_percent: float,
                  results_dir: str):
    
    if not check_uniqueness(prompts):
        raise ValueError("Non-unique rows in prompts")
    split_point = int(len(prompts) * split_percent)
    train = prompts[:split_point]
    validation = prompts[split_point:]
    write_to_csv(Path(results_dir) / 'train.csv', train)
    write_to_csv(Path(results_dir) / 'validation.csv', validation)
