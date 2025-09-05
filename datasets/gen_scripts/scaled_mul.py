import random
import math
from transformers import AutoTokenizer
from utils import is_single_token, compare_token_count, generate_csvs

def scaled_mul(a, b):
    return (a * b) // 10

def get_valid_combos(tokenizer):
    combos = []
    for a in range(10, 100):
        for b in range(10, 100):
            val = scaled_mul(a, b)
            if is_single_token(str(val), tokenizer):
                combos.append((a, b))
    random.shuffle(combos)
    print(f"Valid combos found: {len(combos)}")
    return combos

def corrupt_result(a: int, b: int, tokenizer: AutoTokenizer):
    original_val = scaled_mul(a, b)
    attempts = 0
    while attempts < 1000:
        c = random.randint(10, 99)
        if c != b and compare_token_count(str(b), str(c), tokenizer):
            corrupted_val = scaled_mul(a, c)
            if corrupted_val != original_val and is_single_token(str(corrupted_val), tokenizer):
                return c
        attempts += 1
    return b + 1 if b < 99 else b - 1

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer, symbol:str):
    combos = get_valid_combos(tokenizer)
    prompts = []
    div_counter = 0

    for a, b in combos:
        prompt = prefix_prompt + f'{a}{symbol}{b}='
        c = corrupt_result(a, b, tokenizer)
        corrupted_prompt = prefix_prompt + f'{a}{symbol}{c}='
        length_check = compare_token_count(prompt, corrupted_prompt, tokenizer)
        if not length_check:
            continue  # skip if token counts differ

        answer = scaled_mul(a, b)
        prompts.append([prompt, corrupted_prompt, answer])
        div_counter += 1
        if div_counter >= total_examples:
            break

    if div_counter < total_examples:
        raise ValueError("Not enough examples for dataset")

    return prompts


# Example usage
total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer:'
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer, '*')

generate_csvs(prompts=prompts,
              split_percent=0.9,
              results_dir='../pythia-1.4B/test_set_0/ScaledMul/datasets_csv')
